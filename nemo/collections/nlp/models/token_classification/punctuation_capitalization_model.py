# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from math import ceil
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from tqdm import tqdm

from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss
from nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset import (
    BertPunctuationCapitalizationDataset,
    BertPunctuationCapitalizationInferDataset,
)
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import LogitsType, NeuralType
from nemo.utils import logging

__all__ = ['PunctuationCapitalizationModel']


class PunctuationCapitalizationModel(NLPModel, Exportable):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "punct_logits": NeuralType(('B', 'T', 'C'), LogitsType()),
            "capit_logits": NeuralType(('B', 'T', 'C'), LogitsType()),
        }

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """
        Initializes BERT Punctuation and Capitalization model.
        """
        self.setup_tokenizer(cfg.tokenizer)

        super().__init__(cfg=cfg, trainer=trainer)

        self.bert_model = get_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            config_file=self.register_artifact('language_model.config_file', cfg.language_model.config_file),
            config_dict=OmegaConf.to_container(cfg.language_model.config) if cfg.language_model.config else None,
            checkpoint_file=cfg.language_model.lm_checkpoint,
            vocab_file=self.register_artifact('tokenizer.vocab_file', cfg.tokenizer.vocab_file),
        )

        self.punct_classifier = TokenClassifier(
            hidden_size=self.bert_model.config.hidden_size,
            num_classes=len(self._cfg.punct_label_ids),
            activation=cfg.punct_head.activation,
            log_softmax=False,
            dropout=cfg.punct_head.fc_dropout,
            num_layers=cfg.punct_head.punct_num_fc_layers,
            use_transformer_init=cfg.punct_head.use_transformer_init,
        )

        self.capit_classifier = TokenClassifier(
            hidden_size=self.bert_model.config.hidden_size,
            num_classes=len(self._cfg.capit_label_ids),
            activation=cfg.capit_head.activation,
            log_softmax=False,
            dropout=cfg.capit_head.fc_dropout,
            num_layers=cfg.capit_head.capit_num_fc_layers,
            use_transformer_init=cfg.capit_head.use_transformer_init,
        )

        self.loss = CrossEntropyLoss(logits_ndim=3)
        self.agg_loss = AggregatorLoss(num_inputs=2)

        # setup to track metrics
        self.punct_class_report = ClassificationReport(
            num_classes=len(self._cfg.punct_label_ids),
            label_ids=self._cfg.punct_label_ids,
            mode='macro',
            dist_sync_on_step=True,
        )
        self.capit_class_report = ClassificationReport(
            num_classes=len(self._cfg.capit_label_ids),
            label_ids=self._cfg.capit_label_ids,
            mode='macro',
            dist_sync_on_step=True,
        )

    @typecheck()
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        punct_logits = self.punct_classifier(hidden_states=hidden_states)
        capit_logits = self.capit_classifier(hidden_states=hidden_states)
        return punct_logits, capit_logits

    def _make_step(self, batch):
        input_ids, input_type_ids, input_mask, subtokens_mask, loss_mask, punct_labels, capit_labels = batch
        punct_logits, capit_logits = self(
            input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask
        )

        punct_loss = self.loss(logits=punct_logits, labels=punct_labels, loss_mask=loss_mask)
        capit_loss = self.loss(logits=capit_logits, labels=capit_labels, loss_mask=loss_mask)
        loss = self.agg_loss(loss_1=punct_loss, loss_2=capit_loss)
        return loss, punct_logits, capit_logits

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        loss, _, _ = self._make_step(batch)
        lr = self._optimizer.param_groups[0]['lr']

        self.log('lr', lr, prog_bar=True)
        self.log('train_loss', loss)

        return {'loss': loss, 'lr': lr}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        _, _, _, subtokens_mask, _, punct_labels, capit_labels = batch
        val_loss, punct_logits, capit_logits = self._make_step(batch)

        subtokens_mask = subtokens_mask > 0.5
        punct_preds = torch.argmax(punct_logits, axis=-1)[subtokens_mask]
        punct_labels = punct_labels[subtokens_mask]
        self.punct_class_report.update(punct_preds, punct_labels)

        capit_preds = torch.argmax(capit_logits, axis=-1)[subtokens_mask]
        capit_labels = capit_labels[subtokens_mask]
        self.capit_class_report.update(capit_preds, capit_labels)

        return {
            'val_loss': val_loss,
            'punct_tp': self.punct_class_report.tp,
            'punct_fn': self.punct_class_report.fn,
            'punct_fp': self.punct_class_report.fp,
            'capit_tp': self.capit_class_report.tp,
            'capit_fn': self.capit_class_report.fn,
            'capit_fp': self.capit_class_report.fp,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        _, _, _, subtokens_mask, _, punct_labels, capit_labels = batch
        test_loss, punct_logits, capit_logits = self._make_step(batch)

        subtokens_mask = subtokens_mask > 0.5
        punct_preds = torch.argmax(punct_logits, axis=-1)[subtokens_mask]
        punct_labels = punct_labels[subtokens_mask]
        self.punct_class_report.update(punct_preds, punct_labels)

        capit_preds = torch.argmax(capit_logits, axis=-1)[subtokens_mask]
        capit_labels = capit_labels[subtokens_mask]
        self.capit_class_report.update(capit_preds, capit_labels)

        return {
            'test_loss': test_loss,
            'punct_tp': self.punct_class_report.tp,
            'punct_fn': self.punct_class_report.fn,
            'punct_fp': self.punct_class_report.fp,
            'capit_tp': self.capit_class_report.tp,
            'capit_fn': self.capit_class_report.fn,
            'capit_fp': self.capit_class_report.fp,
        }

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        """
        Called at the end of validation to aggregate outputs.
        outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # calculate metrics and log classification report for Punctuation task
        punct_precision, punct_recall, punct_f1, punct_report = self.punct_class_report.compute()
        logging.info(f'Punctuation report: {punct_report}')

        # calculate metrics and log classification report for Capitalization task
        capit_precision, capit_recall, capit_f1, capit_report = self.capit_class_report.compute()
        logging.info(f'Capitalization report: {capit_report}')

        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('punct_precision', punct_precision)
        self.log('punct_f1', punct_f1)
        self.log('punct_recall', punct_recall)
        self.log('capit_precision', capit_precision)
        self.log('capit_f1', capit_f1)
        self.log('capit_recall', capit_recall)

        self.punct_class_report.reset()
        self.capit_class_report.reset()

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        """
            Called at the end of test to aggregate outputs.
            outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        # calculate metrics and log classification report for Punctuation task
        punct_precision, punct_recall, punct_f1, punct_report = self.punct_class_report.compute()
        logging.info(f'Punctuation report: {punct_report}')

        # calculate metrics and log classification report for Capitalization task
        capit_precision, capit_recall, capit_f1, capit_report = self.capit_class_report.compute()
        logging.info(f'Capitalization report: {capit_report}')

        self.log('test_loss', avg_loss, prog_bar=True)
        self.log('punct_precision', punct_precision)
        self.log('punct_f1', punct_f1)
        self.log('punct_recall', punct_recall)
        self.log('capit_precision', capit_precision)
        self.log('capit_f1', capit_f1)
        self.log('capit_recall', capit_recall)

    def update_data_dir(self, data_dir: str) -> None:
        """
        Update data directory

        Args:
            data_dir: path to data directory
        """
        if os.path.exists(data_dir):
            logging.info(f'Setting model.dataset.data_dir to {data_dir}.')
            self._cfg.dataset.data_dir = data_dir
        else:
            raise ValueError(f'{data_dir} not found')

    def setup_training_data(self, train_data_config: Optional[DictConfig] = None):
        """Setup training data"""
        if train_data_config is None:
            train_data_config = self._cfg.train_ds

        # for older(pre - 1.0.0.b3) configs compatibility
        if not hasattr(self._cfg, "class_labels") or self._cfg.class_labels is None:
            OmegaConf.set_struct(self._cfg, False)
            self._cfg.class_labels = {}
            self._cfg.class_labels = OmegaConf.create(
                {'punct_labels_file': 'punct_label_ids.csv', 'capit_labels_file': 'capit_label_ids.csv'}
            )

        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            self.register_artifact('class_labels.punct_labels_file', self._train_dl.dataset.punct_label_ids_file)
            self.register_artifact('class_labels.capit_labels_file', self._train_dl.dataset.capit_label_ids_file)

            # save label maps to the config
            self._cfg.punct_label_ids = OmegaConf.create(self._train_dl.dataset.punct_label_ids)
            self._cfg.capit_label_ids = OmegaConf.create(self._train_dl.dataset.capit_label_ids)

    def setup_validation_data(self, val_data_config: Optional[Dict] = None):
        """
        Setup validaton data

        val_data_config: validation data config
        """
        if val_data_config is None:
            val_data_config = self._cfg.validation_ds

        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Dict] = None):
        if test_data_config is None:
            test_data_config = self._cfg.test_ds
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    def _setup_dataloader_from_config(self, cfg: DictConfig):
        # use data_dir specified in the ds_item to run evaluation on multiple datasets
        if 'ds_item' in cfg and cfg.ds_item is not None:
            data_dir = cfg.ds_item
        else:
            data_dir = self._cfg.dataset.data_dir

        text_file = os.path.join(data_dir, cfg.text_file)
        label_file = os.path.join(data_dir, cfg.labels_file)

        dataset = BertPunctuationCapitalizationDataset(
            tokenizer=self.tokenizer,
            text_file=text_file,
            label_file=label_file,
            pad_label=self._cfg.dataset.pad_label,
            punct_label_ids=self._cfg.punct_label_ids,
            capit_label_ids=self._cfg.capit_label_ids,
            max_seq_length=self._cfg.dataset.max_seq_length,
            ignore_extra_tokens=self._cfg.dataset.ignore_extra_tokens,
            ignore_start_end=self._cfg.dataset.ignore_start_end,
            use_cache=self._cfg.dataset.use_cache,
            num_samples=cfg.num_samples,
            punct_label_ids_file=self._cfg.class_labels.punct_labels_file
            if 'class_labels' in self._cfg
            else 'punct_label_ids.csv',
            capit_label_ids_file=self._cfg.class_labels.capit_labels_file
            if 'class_labels' in self._cfg
            else 'capit_label_ids.csv',
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=self._cfg.dataset.num_workers,
            pin_memory=self._cfg.dataset.pin_memory,
            drop_last=self._cfg.dataset.drop_last,
        )

    def _setup_infer_dataloader(
        self, queries: List[str], batch_size: int, max_seq_length: int, step: int, margin: int,
    ) -> torch.utils.data.DataLoader:
        """
        Setup function for a infer data loader.

        Args:
            model: a ``PunctuationCapitalizationModel`` instance for which data loader is created.
            queries: lower cased text without punctuation
            batch_size: batch size to use during inference
            max_seq_length: length of segments into which queries are split. ``max_seq_length`` includes ``[CLS]`` and
                ``[SEP]`` so every segment contains at most ``max_seq_length-2`` tokens from input a query.
            step: number of tokens by which a segment is offset to a previous segment. Parameter ``step`` cannot be greater
                than ``max_seq_length-2``.
            margin: number of tokens near the edge of a segment which label probabilities are not used in final prediction
                computation.
        Returns:
            A pytorch DataLoader.
        """
        if max_seq_length is None:
            max_seq_length = self._cfg.dataset.max_seq_length
        if step is None:
            step = self._cfg.dataset.step
        if margin is None:
            margin = self._cfg.dataset.margin

        dataset = BertPunctuationCapitalizationInferDataset(
            tokenizer=self.tokenizer, queries=queries, max_seq_length=max_seq_length, step=step, margin=margin
        )
        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self._cfg.dataset.num_workers,
            pin_memory=self._cfg.dataset.pin_memory,
            drop_last=False,
        )

    @staticmethod
    def _remove_margins(tensor, margin_size, keep_left, keep_right):
        tensor = tensor.detach().clone()
        if not keep_left:
            tensor = tensor[margin_size + 1 :]  # remove left margin and CLS token
        if not keep_right:
            tensor = tensor[: tensor.shape[0] - margin_size - 1]  # remove right margin and SEP token
        return tensor

    def _transform_logit_to_prob_and_remove_margins_and_extract_word_probs(
        self,
        punct_logits: torch.Tensor,
        capit_logits: torch.Tensor,
        subtokens_mask: torch.Tensor,
        start_word_ids: Tuple[int],
        margin: int,
        is_first: Tuple[bool],
        is_last: Tuple[bool],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """
        Applies softmax to get punctuation and capitalization probabilities, applies ``subtokens_mask`` to extract
        probabilities for words from probabilities for tokens, removes ``margin`` probabilities near edges of a segment.
        Left margin of the first segment in a query and right margin of the last segment in a query are not removed.
        Calculates new ``start_word_ids`` taking into the account the margins. If the left margin of a segment is removed
        corresponding start word index is increased by number of words (number of nonzero values in corresponding
        ``subtokens_mask``) in the margin.
        Args:
            punct_logits: a float tensor of shape ``[batch_size, segment_length, number_of_punctuation_labels]``
            capit_logits: a float tensor of shape ``[batch_size, segment_length, number_of_capitalization_labels]``
            subtokens_mask: a float tensor of shape ``[batch_size, segment_length]``
            start_word_ids: indices of segment first words in a query
            margin: number of tokens near edges of a segment which probabilities are discarded
            is_first: is segment the first segment in a query
            is_last: is segment the last segment in a query
        Returns:
            b_punct_probs: list containing ``batch_size`` numpy arrays. The numpy arrays have shapes
                ``[number_of_word_in_this_segment, number_of_punctuation_labels]``. Word punctuation probabilities for
                segments in the batch.
            b_capit_probs: list containing ``batch_size`` numpy arrays. The numpy arrays have shapes
                ``[number_of_word_in_this_segment, number_of_capitalization_labels]``. Word capitalization probabilities for
                segments in the batch.
            new_start_word_ids: indices of segment first words in a query after margin removal
        """
        new_start_word_ids = list(start_word_ids)
        subtokens_mask = subtokens_mask > 0.5
        b_punct_probs, b_capit_probs = [], []
        for i, (first, last, pl, cl, stm) in enumerate(
            zip(is_first, is_last, punct_logits, capit_logits, subtokens_mask)
        ):
            if not first:
                new_start_word_ids[i] += torch.count_nonzero(stm[: margin + 1]).numpy()  # + 1 is for [CLS] token
            stm = self._remove_margins(stm, margin, keep_left=first, keep_right=last)
            for b_probs, logits in [(b_punct_probs, pl), (b_capit_probs, cl)]:
                p = torch.nn.functional.softmax(
                    self._remove_margins(logits, margin, keep_left=first, keep_right=last)[stm], dim=-1,
                )
                b_probs.append(p.detach().cpu().numpy())
        return b_punct_probs, b_capit_probs, new_start_word_ids

    @staticmethod
    def _move_acc_probs_to_token_preds(
        pred: List[int], acc_prob: np.ndarray, number_of_probs_to_move: int
    ) -> Tuple[List[int], np.ndarray]:
        """
        ``number_of_probs_to_move`` rows in the beginning are removed from ``acc_prob``. From every remove row the label
        with the largest probability is selected and appended to ``pred``.
        Args:
            pred: list with ready label indices for a query
            acc_prob: numpy array of shape ``[number_of_words_for_which_probabilities_are_accumulated, number_of_labels]``
            number_of_probs_to_move: int
        Returns:
            pred: list with ready label indices for a query
            acc_prob: numpy array of shape
                ``[number_of_words_for_which_probabilities_are_accumulated - number_of_probs_to_move, number_of_labels]``
        """
        if number_of_probs_to_move > acc_prob.shape[0]:
            raise ValueError(
                f"Not enough accumulated probabilities. Number_of_probs_to_move={number_of_probs_to_move} "
                f"acc_prob.shape={acc_prob.shape}"
            )
        if number_of_probs_to_move > 0:
            pred = pred + list(np.argmax(acc_prob[:number_of_probs_to_move], axis=-1))
        acc_prob = acc_prob[number_of_probs_to_move:]
        return pred, acc_prob

    @staticmethod
    def _update_accumulated_probabilities(acc_prob: np.ndarray, update: np.ndarray) -> np.ndarray:
        """
        Args:
            acc_prob: numpy array of shape ``[A, L]``
            update: numpy array of shape ``[A + N, L]``
        Returns:
            numpy array of shape ``[A + N, L]``
        """
        acc_prob = np.concatenate([acc_prob * update[: acc_prob.shape[0]], update[acc_prob.shape[0] :]], axis=0)
        return acc_prob

    def apply_punct_capit_predictions(self, query: str, punct_preds: List[int], capit_preds: List[int]) -> str:
        """
        Restores punctuation and capitalization in ``query``.
        Args:
            query: a string without punctuation and capitalization
            punct_preds: ids of predicted punctuation labels
            capit_preds: ids of predicted capitalization labels
        Returns:
            a query with restored punctuation and capitalization
        """
        query = query.strip().split()
        assert len(query) == len(
            punct_preds
        ), f"len(query)={len(query)} len(punct_preds)={len(punct_preds)}, query[:30]={query[:30]}"
        assert len(query) == len(
            capit_preds
        ), f"len(query)={len(query)} len(capit_preds)={len(capit_preds)}, query[:30]={query[:30]}"
        punct_ids_to_labels = {v: k for k, v in self._cfg.punct_label_ids.items()}
        capit_ids_to_labels = {v: k for k, v in self._cfg.capit_label_ids.items()}
        query_with_punct_and_capit = ''
        for j, word in enumerate(query):
            punct_label = punct_ids_to_labels[punct_preds[j]]
            capit_label = capit_ids_to_labels[capit_preds[j]]

            if capit_label != self._cfg.dataset.pad_label:
                word = word.capitalize()
            query_with_punct_and_capit += word
            if punct_label != self._cfg.dataset.pad_label:
                query_with_punct_and_capit += punct_label
            query_with_punct_and_capit += ' '
        return query_with_punct_and_capit[:-1]

    def get_labels(self, punct_preds: List[int], capit_preds: List[int]) -> str:
        """
        Returns punctuation and capitalization labels in NeMo format (see https://docs.nvidia.com/deeplearning/nemo/
        user-guide/docs/en/main/nlp/punctuation_and_capitalization.html#nemo-data-format).
        Args:
            punct_preds: ids of predicted punctuation labels
            capit_preds: ids of predicted capitalization labels
        Returns:
            labels in NeMo format
        """
        assert len(capit_preds) == len(
            punct_preds
        ), f"len(capit_preds)={len(capit_preds)} len(punct_preds)={len(punct_preds)}"
        punct_ids_to_labels = {v: k for k, v in self._cfg.punct_label_ids.items()}
        capit_ids_to_labels = {v: k for k, v in self._cfg.capit_label_ids.items()}
        result = ''
        for capit_label, punct_label in zip(capit_preds, punct_preds):
            punct_label = punct_ids_to_labels[punct_label]
            capit_label = capit_ids_to_labels[capit_label]
            result += punct_label + capit_label + ' '
        return result[:-1]

    def add_punctuation_capitalization(
        self,
        queries: List[str],
        batch_size: int = None,
        max_seq_length: int = 64,
        step: int = 8,
        margin: int = 16,
        return_labels: bool = False,
    ) -> List[str]:
        """
        Adds punctuation and capitalization to the queries. Use this method for inference.

        Parameters ``max_seq_length``, ``step``, ``margin`` are for controlling the way queries are split into segments
        which then processed by the model. Parameter ``max_seq_length`` is a length of a segment after tokenization
        including special tokens [CLS] in the beginning and [SEP] in the end of a segment. Parameter ``step`` is shift
        between consequent segments. Parameter ``margin`` is used to exclude negative effect of subtokens near
        borders of segments which have only one side context.

        If segments overlap, probabilities of overlapping predictions are multiplied and then the label with
        corresponding to the maximum probability is selected.

        Args:
            queries: lower cased text without punctuation
            batch_size: batch size to use during inference
            max_seq_length: maximum sequence length of segment after tokenization.
            step: relative shift of consequent segments into which long queries are split. Long queries are split into
                segments which can overlap. Parameter ``step`` controls such overlapping. Imagine that queries are
                tokenized into characters, ``max_seq_length=5``, and ``step=2``. In such a case query "hello" is
                tokenized into segments ``[['[CLS]', 'h', 'e', 'l', '[SEP]'], ['[CLS]', 'l', 'l', 'o', '[SEP]']]``.
            margin: number of subtokens in the beginning and the end of segments which are not used for prediction
                computation. The first segment does not have left margin and the last segment does not have right
                margin. For example, if input sequence is tokenized into characters, ``max_seq_length=5``,
                ``step=1``, and ``margin=1``, then query "hello" will be tokenized into segments
                ``[['[CLS]', 'h', 'e', 'l', '[SEP]'], ['[CLS]', 'e', 'l', 'l', '[SEP]'],
                ['[CLS]', 'l', 'l', 'o', '[SEP]']]``. These segments are passed to the model. Before final predictions
                computation, margins are removed. In the next list, subtokens which logits are not used for final
                predictions computation are marked with asterisk: ``[['[CLS]'*, 'h', 'e', 'l'*, '[SEP]'*],
                ['[CLS]'*, 'e'*, 'l', 'l'*, '[SEP]'*], ['[CLS]'*, 'l'*, 'l', 'o', '[SEP]'*]]``.
            return_labels: whether to return labels in NeMo format (see https://docs.nvidia.com/deeplearning/nemo/
                user-guide/docs/en/main/nlp/punctuation_and_capitalization.html#nemo-data-format) instead of queries
                with restored punctuation and capitalization.
        Returns:
            result: text with added capitalization and punctuation or punctuation and capitalization labels
        """
        if len(queries) == 0:
            return []
        if batch_size is None:
            batch_size = len(queries)
            logging.info(f'Using batch size {batch_size} for inference')
        result: List[str] = []
        mode = self.training
        try:
            self.eval()
            infer_datalayer = self._setup_infer_dataloader(queries, batch_size, max_seq_length, step, margin)
            # Predicted labels for queries. List of labels for every query
            all_punct_preds: List[List[int]] = [[] for _ in queries]
            all_capit_preds: List[List[int]] = [[] for _ in queries]
            # Accumulated probabilities (or product of probabilities acquired from different segments) of punctuation
            # and capitalization. Probabilities for words in a query are extracted using `subtokens_mask`. Probabilities
            # for newly processed words are appended to the accumulated probabilities. If probabilities for a word are
            # already present in `acc_probs`, old probabilities are replaced with a product of old probabilities
            # and probabilities acquired from new segment. Segments are processed in an order they appear in an
            # input query. When all segments with a word are processed, a label with the highest probability
            # (or product of probabilities) is chosen and appended to an appropriate list in `all_preds`. After adding
            # prediction to `all_preds`, probabilities for a word are removed from `acc_probs`.
            acc_punct_probs: List[Optional[np.ndarray]] = [None for _ in queries]
            acc_capit_probs: List[Optional[np.ndarray]] = [None for _ in queries]
            d = self.device
            for batch_i, batch in tqdm(
                enumerate(infer_datalayer), total=ceil(len(infer_datalayer.dataset) / batch_size), unit="batch"
            ):
                inp_ids, inp_type_ids, inp_mask, subtokens_mask, start_word_ids, query_ids, is_first, is_last = batch
                punct_logits, capit_logits = self.forward(
                    input_ids=inp_ids.to(d), token_type_ids=inp_type_ids.to(d), attention_mask=inp_mask.to(d),
                )
                _res = self._transform_logit_to_prob_and_remove_margins_and_extract_word_probs(
                    punct_logits, capit_logits, subtokens_mask, start_word_ids, margin, is_first, is_last
                )
                punct_probs, capit_probs, start_word_ids = _res
                for i, (q_i, start_word_id, bpp_i, bcp_i) in enumerate(
                    zip(query_ids, start_word_ids, punct_probs, capit_probs)
                ):
                    for all_preds, acc_probs, b_probs_i in [
                        (all_punct_preds, acc_punct_probs, bpp_i),
                        (all_capit_preds, acc_capit_probs, bcp_i),
                    ]:
                        if acc_probs[q_i] is None:
                            acc_probs[q_i] = b_probs_i
                        else:
                            all_preds[q_i], acc_probs[q_i] = self._move_acc_probs_to_token_preds(
                                all_preds[q_i], acc_probs[q_i], start_word_id - len(all_preds[q_i]),
                            )
                            acc_probs[q_i] = self._update_accumulated_probabilities(acc_probs[q_i], b_probs_i)
            for all_preds, acc_probs in [(all_punct_preds, acc_punct_probs), (all_capit_preds, acc_capit_probs)]:
                for q_i, (pred, prob) in enumerate(zip(all_preds, acc_probs)):
                    if prob is not None:
                        all_preds[q_i], acc_probs[q_i] = self._move_acc_probs_to_token_preds(pred, prob, len(prob))
            for i, query in enumerate(queries):
                result.append(
                    self.get_labels(all_punct_preds[i], all_capit_preds[i])
                    if return_labels
                    else self.apply_punct_capit_predictions(query, all_punct_preds[i], all_capit_preds[i])
                )
        finally:
            # set mode back to its original value
            self.train(mode=mode)
        return result

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        result = []
        result.append(
            PretrainedModelInfo(
                pretrained_model_name="punctuation_en_bert",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/punctuation_en_bert/versions/1.0.0rc1/files/punctuation_en_bert.nemo",
                description="The model was trained with NeMo BERT base uncased checkpoint on a subset of data from the following sources: Tatoeba sentences, books from Project Gutenberg, Fisher transcripts.",
            )
        )
        result.append(
            PretrainedModelInfo(
                pretrained_model_name="punctuation_en_distilbert",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/punctuation_en_distilbert/versions/1.0.0rc1/files/punctuation_en_distilbert.nemo",
                description="The model was trained with DiltilBERT base uncased checkpoint from HuggingFace on a subset of data from the following sources: Tatoeba sentences, books from Project Gutenberg, Fisher transcripts.",
            )
        )
        return result

    @property
    def input_module(self):
        return self.bert_model

    @property
    def output_module(self):
        return self
