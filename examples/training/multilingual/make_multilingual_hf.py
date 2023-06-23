import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Union, Any, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from numpy import ndarray
from transformers import Trainer, HfArgumentParser, TrainingArguments, EvalPrediction

from sentence_transformers import LoggingHandler, SentenceTransformer, ParallelSentencesDataset
from sentence_transformers.models import Transformer, Pooling, Dense, Normalize
from sentence_transformers.util import pytorch_cos_sim


class SentenceTransformerMultilingualDataCollator:
    def __init__(self, model):
        self.model = model

    def __call__(self, batch):
        """
        Adopted from
        https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py#L529
        Transforms a batch from a SmartBatchingDataset to a batched input expected by HuggingFace Transformer
        Here, batch is a list of tuples: [(tokens, label), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch dict of tensors for the model
        """
        num_texts = len(batch[0].texts)  # num of languages
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

            labels.append(example.label)
        if isinstance(labels[0], ndarray):
            labels = np.array(labels)
        labels = torch.tensor(labels)

        texts_flatten = [t for by_lang in texts for t in by_lang]
        # assumed each sentence has the same num of language variants
        multilingual_sentence_features = self.model.tokenize(texts_flatten)
        labels_repeated = torch.repeat_interleave(labels, num_texts, 0)

        multilingual_sentence_features.update(labels=labels_repeated, return_loss=torch.as_tensor([True, ]))

        return multilingual_sentence_features


class Evaluator(ABC):
    @abstractmethod
    def __call__(self, eval_preds: EvalPrediction) -> Dict[str, float]:
        raise NotImplementedError


class SentenceTransformerMSEEvaluator(Evaluator):
    def __call__(self, eval_preds: EvalPrediction) -> Dict[str, float]:
        source_embeddings = eval_preds.predictions
        target_embeddings = eval_preds.label_ids
        mse = ((source_embeddings - target_embeddings) ** 2).mean()
        mse *= 100

        if dist.get_rank() == 0:
            logger.info("MSE evaluation (lower = better)")
            logger.info("MSE (*100):\t{:4f}".format(mse))

        return {"mse": -mse}


class SentenceTransformerTranslationEvaluator(Evaluator):
    def __call__(self, eval_preds: EvalPrediction) -> Dict[str, float]:
        source_embeddings = eval_preds.predictions
        target_embeddings = eval_preds.label_ids
        cos_sims = pytorch_cos_sim(source_embeddings, target_embeddings).detach().cpu().numpy()

        correct_src2trg = 0
        correct_trg2src = 0

        for i in range(len(cos_sims)):
            max_idx = np.argmax(cos_sims[i])

            if i == max_idx:
                correct_src2trg += 1

        cos_sims = cos_sims.T
        for i in range(len(cos_sims)):
            max_idx = np.argmax(cos_sims[i])
            if i == max_idx:
                correct_trg2src += 1

        acc_src2trg = correct_src2trg / len(cos_sims)
        acc_trg2src = correct_trg2src / len(cos_sims)

        if dist.get_rank() == 0:
            logger.info("Accuracy src2trg: {:.2f}".format(acc_src2trg * 100))
            logger.info("Accuracy trg2src: {:.2f}".format(acc_trg2src * 100))

        return {
            "mean_translation_accuracy": (acc_src2trg + acc_trg2src) / 2
        }


class SequentialEvaluator(Evaluator):
    def __init__(self, evaluators: List[Evaluator]):
        self.evaluators = evaluators

    def __call__(self, eval_preds: EvalPrediction) -> Dict[str, float]:
        metrics = {}
        for e in self.evaluators:
            metrics.update(e(eval_preds))
        return metrics


class SentenceTransformerMultilingualTrainer(Trainer):
    model: SentenceTransformer

    def __init__(self, evaluator: Evaluator = None, **kwargs):
        super().__init__(**kwargs)
        self.evaluator = evaluator
        self.compute_metrics = self.compute_metrics_impl

    def compute_loss(self, model: SentenceTransformer, inputs, return_outputs=False):
        outputs = model(inputs)
        loss = F.mse_loss(outputs['sentence_embedding'], outputs['labels'])
        return (loss, outputs) if return_outputs else loss

    def compute_metrics_impl(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        return self.evaluator(eval_pred)

    def prediction_step(
        self,
        model: SentenceTransformer,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
        return loss, outputs['sentence_embedding'], outputs["labels"]


def parse_args():
    parser = HfArgumentParser(TrainingArguments)
    parser.set_defaults(include_inputs_for_metrics=True)
    parser.set_defaults(prediction_loss_only=False)
    parser.add_argument("--teacher", type=str, default="sentence-transformers/gtr-t5-base")
    parser.add_argument("--student", type=str, default="xlm-roberta-base")
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Maximum length (number of word pieces) for input sequences')
    parser.add_argument('--teacher_inference_batch_size', type=int, default=64, help='Batch size at teacher inference')
    parser.add_argument('--max_sentences_per_trainfile', type=int, default=500000,
                        help='Maximum number of parallel sentences for training')
    parser.add_argument('--max_sentences_per_testfile', type=int, default=500000,
                        help='Maximum number of parallel sentences for training')
    parser.add_argument('--train_max_sentence_length', type=int, default=250,
                        help='Maximum length (characters) for parallel training sentences')
    parser.add_argument('--test_max_sentence_length', type=int, default=250,
                        help='Maximum length (characters) for parallel training sentences')
    parser.add_argument("--train_files", nargs="+", type=str, help="parallel sentence tsv for training")
    parser.add_argument("--test_files", nargs="+", type=str, help="parallel sentence tsv for testing")
    parser.add_argument("--t5_student", action="store_true", help="whether add a projection layer at the end")
    parser.add_argument("--t5_student_d_model", type=int, default=None, help="out channels of the projection layer")
    return parser.parse_args_into_dataclasses()


if __name__ == "__main__":
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO,
                        handlers=[LoggingHandler()])
    logger = logging.getLogger(__name__)
    trainer_args, args = parse_args()

    logger.info(f"Load teacher model: {args.teacher}")
    teacher_model = SentenceTransformer(args.teacher)
    teacher_model.eval()

    logger.info(f"Create student model: {args.student}")
    word_embedding_model = Transformer(args.student, max_seq_length=args.max_seq_length)
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension())
    _modules = [word_embedding_model, pooling_model]
    if args.t5_student:
        assert args.t5_student_d_model is not None, f"Invalid value for t5_student_d_model: {args.t5_student_d_model}"
        _modules.append(Dense(
            word_embedding_model.get_word_embedding_dimension(), args.t5_student_d_model,
            bias=False, activation_function=torch.nn.modules.linear.Identity()
        ))
        _modules.append(Normalize())
    student_model = SentenceTransformer(modules=_modules)

    train_data = ParallelSentencesDataset(student_model=student_model, teacher_model=teacher_model,
                                          batch_size=args.teacher_inference_batch_size, use_embedding_cache=True)

    eval_data = ParallelSentencesDataset(student_model=student_model, teacher_model=teacher_model,
                                         batch_size=args.teacher_inference_batch_size, use_embedding_cache=True)

    data_collator = SentenceTransformerMultilingualDataCollator(student_model)

    for train_file in args.train_files:
        train_data.load_data(train_file, max_sentences=args.max_sentences_per_trainfile,
                             max_sentence_length=args.train_max_sentence_length)
    for test_file in args.test_files:
        eval_data.load_data(test_file, max_sentences=args.max_sentences_per_testfile,
                            max_sentence_length=args.test_max_sentence_length)

    trainer = SentenceTransformerMultilingualTrainer(
        model=student_model,
        args=trainer_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        compute_metrics=SentenceTransformerMultilingualTrainer.compute_metrics_impl,
        evaluator=SequentialEvaluator([
            SentenceTransformerMSEEvaluator(),
            SentenceTransformerTranslationEvaluator()
        ])
    )

    trainer.train()  # save model?
    trainer.model.save(os.path.join(trainer.args.output_dir, "make-multilingual-hf"))
