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
from typing import Dict, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss
from nemo.collections.common.tokenizers.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.data.data_utils.data_preprocessing import get_labels_to_labels_id_mapping
from nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset import (
    BertPunctuationCapitalizationDataset,
)
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.common_utils import get_pretrained_lm_model
from nemo.core.classes import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import LogitsType, NeuralType
from nemo.utils import logging

__all__ = ['PunctuationCapitalizationModel']


class PunctuationCapitalizationModel(ModelPT):
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
        self.data_dir = cfg.dataset.data_dir
        self.tokenizer = get_tokenizer(
            tokenizer_name=cfg.language_model.tokenizer,
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            vocab_file=self.register_artifact(
                config_path='language_model.vocab_file', src=cfg.language_model.vocab_file
            ),
            tokenizer_model=self.register_artifact(
                config_path='language_model.tokenizer_model', src=cfg.language_model.tokenizer_model
            ),
            do_lower_case=cfg.language_model.do_lower_case,
        )

        super().__init__(cfg=cfg, trainer=trainer)

        self.bert_model = get_pretrained_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            config_file=cfg.language_model.bert_config,
            checkpoint_file=cfg.language_model.bert_checkpoint,
        )

        self.hidden_size = self.bert_model.config.hidden_size

        self.punct_classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=len(self.punct_label_ids),
            activation=cfg.punct_head.activation,
            log_softmax=cfg.punct_head.log_softmax,
            dropout=cfg.punct_head.fc_dropout,
            num_layers=cfg.punct_head.punct_num_fc_layers,
            use_transformer_init=cfg.punct_head.use_transformer_init,
        )

        self.capit_classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=len(self.capit_label_ids),
            activation=cfg.capit_head.activation,
            log_softmax=cfg.capit_head.log_softmax,
            dropout=cfg.capit_head.fc_dropout,
            num_layers=cfg.capit_head.capit_num_fc_layers,
            use_transformer_init=cfg.capit_head.use_transformer_init,
        )

        self.loss = CrossEntropyLoss(logits_ndim=3)
        self.agg_loss = AggregatorLoss(num_inputs=2)

        # setup to track metrics
        self.punct_class_report = ClassificationReport(len(self.punct_label_ids), label_ids=self.punct_label_ids)
        self.capit_class_report = ClassificationReport(len(self.capit_label_ids), label_ids=self.capit_label_ids)

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

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, punct_labels, capit_labels = batch
        punct_logits, capit_logits = self(
            input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask
        )

        punct_loss = self.loss(logits=punct_logits, labels=punct_labels, loss_mask=loss_mask)
        capit_loss = self.loss(logits=capit_logits, labels=capit_labels, loss_mask=loss_mask)
        loss = self.agg_loss(loss_1=punct_loss, loss_2=capit_loss)
        tensorboard_logs = {'train_loss': loss, 'lr': self._optimizer.param_groups[0]['lr']}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, punct_labels, capit_labels = batch
        punct_logits, capit_logits = self(
            input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask
        )

        punct_loss = self.loss(logits=punct_logits, labels=punct_labels, loss_mask=loss_mask)
        capit_loss = self.loss(logits=capit_logits, labels=capit_labels, loss_mask=loss_mask)
        val_loss = self.agg_loss(loss_1=punct_loss, loss_2=capit_loss)

        subtokens_mask = subtokens_mask > 0.5

        punct_preds = torch.argmax(punct_logits, axis=-1)[subtokens_mask]
        punct_labels = punct_labels[subtokens_mask]
        punct_tp, punct_fp, punct_fn = self.punct_class_report(punct_preds, punct_labels)

        capit_preds = torch.argmax(capit_logits, axis=-1)[subtokens_mask]
        capit_labels = capit_labels[subtokens_mask]
        capit_tp, capit_fp, capit_fn = self.capit_class_report(capit_preds, capit_labels)
        tensorboard_logs = {
            'val_loss': val_loss,
            'punct_tp': punct_tp,
            'punct_fn': punct_fn,
            'punct_fp': punct_fp,
            'capit_tp': capit_tp,
            'capit_fn': capit_fn,
            'capit_fp': capit_fp,
        }

        return {
            'val_loss': val_loss,
            'log': tensorboard_logs,
        }

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        """
        Called at the end of validation to aggregate outputs.
        outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # calculate metrics and log classification report for Punctuation task
        punct_tp = torch.sum(torch.stack([x['log']['punct_tp'] for x in outputs]), 0)
        punct_fn = torch.sum(torch.stack([x['log']['punct_fn'] for x in outputs]), 0)
        punct_fp = torch.sum(torch.stack([x['log']['punct_fp'] for x in outputs]), 0)
        punct_precision, punct_recall, punct_f1 = self.punct_class_report.get_precision_recall_f1(
            punct_tp, punct_fn, punct_fp, mode='macro'
        )

        # calculate metrics and log classification report for Capitalization task
        capit_tp = torch.sum(torch.stack([x['log']['capit_tp'] for x in outputs]), 0)
        capit_fn = torch.sum(torch.stack([x['log']['capit_fn'] for x in outputs]), 0)
        capit_fp = torch.sum(torch.stack([x['log']['capit_fp'] for x in outputs]), 0)
        capit_precision, capit_recall, capit_f1 = self.capit_class_report.get_precision_recall_f1(
            capit_tp, capit_fn, capit_fp, mode='macro'
        )
        tensorboard_logs = {
            'validation_loss': avg_loss,
            'punct_precision': punct_precision,
            'punct_f1': punct_f1,
            'punct_recall': punct_recall,
            'capit_precision': capit_precision,
            'capit_f1': capit_f1,
            'capit_recall': capit_recall,
        }
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[Dict]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Dict]):
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    def _setup_dataloader_from_config(self, cfg: DictConfig):
        if cfg.prefix == 'train' or 'ds_item' not in cfg or cfg.ds_item is None:
            text_file = os.path.join(self.data_dir, 'text_' + cfg.prefix + '.txt')
            label_file = os.path.join(self.data_dir, 'labels_' + cfg.prefix + '.txt')
        else:
            # use data_dir specified in the ds_item to run evaluation on multiple datasets
            text_file = os.path.join(cfg.ds_item, 'text_' + cfg.prefix + '.txt')
            label_file = os.path.join(cfg.ds_item, 'labels_' + cfg.prefix + '.txt')
        if not (os.path.exists(text_file) and os.path.exists(label_file)):
            raise FileNotFoundError(
                f'{text_file} or {label_file} not found. The data should be splitted into 2 files: text.txt and \
                labels.txt. Each line of the text.txt file contains text sequences, where words are separated with \
                spaces. The labels.txt file contains corresponding labels for each word in text.txt, the labels are \
                separated with spaces. Each line of the files should follow the format:  \
                   [WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt) and \
                   [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt).'
            )

        if cfg.prefix == 'train':
            punct_label_ids_file = os.path.join(self.data_dir, 'punct_label_ids.csv')
            capit_label_ids_file = os.path.join(self.data_dir, 'capit_label_ids.csv')

            if (
                self._cfg.dataset.use_cache
                and os.path.exists(punct_label_ids_file)
                and os.path.exists(capit_label_ids_file)
            ):
                logging.info(f'Restoring punct_label_ids from {punct_label_ids_file}')
                self.punct_label_ids = get_labels_to_labels_id_mapping(punct_label_ids_file)
                logging.info(f'Restoring capit_label_ids from {capit_label_ids_file}')
                self.capit_label_ids = get_labels_to_labels_id_mapping(capit_label_ids_file)
            else:
                self.punct_label_ids = None
                self.capit_label_ids = None

        dataset = BertPunctuationCapitalizationDataset(
            tokenizer=self.tokenizer,
            text_file=text_file,
            label_file=label_file,
            pad_label=self._cfg.dataset.pad_label,
            punct_label_ids=self.punct_label_ids,
            capit_label_ids=self.capit_label_ids,
            max_seq_length=self._cfg.dataset.max_seq_length,
            ignore_extra_tokens=self._cfg.dataset.ignore_extra_tokens,
            ignore_start_end=self._cfg.dataset.ignore_start_end,
            use_cache=self._cfg.dataset.use_cache,
            num_samples=cfg.num_samples,
        )
        if cfg.prefix == 'train':
            self.punct_label_ids = dataset.punct_label_ids
            self.capit_label_ids = dataset.capit_label_ids
            self.register_artifact('punct_label_ids.csv', punct_label_ids_file)
            self.register_artifact('capit_label_ids.csv', capit_label_ids_file)

        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=self._cfg.dataset.num_workers,
            pin_memory=self._cfg.dataset.pin_memory,
            drop_last=self._cfg.dataset.drop_last,
        )

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass
