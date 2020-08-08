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
import pickle
from typing import Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.common.tokenizers.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.data.token_classification.token_classification_dataset import BertTokenClassificationDataset
from nemo.collections.nlp.data.token_classification.token_classification_descriptor import TokenClassificationDataDesc
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.common_utils import get_pretrained_lm_model
from nemo.core.classes import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import NeuralType
from nemo.utils import logging

__all__ = ['TokenClassificationModel']


class TokenClassificationModel(ModelPT):
    """Token Classification Model with BERT, applicable for tasks such as Named Entity Recognition"""

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.classifier.output_types

    @classmethod
    def update_config_with_specific_artifacts(cls, config: OmegaConf, artifacts_dir: str) -> OmegaConf:
        config.data_desc_pickle = os.path.join(artifacts_dir, config.data_desc_pickle)
        return config

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """Initializes Token Classification Model."""

        self.data_dir = cfg.dataset.data_dir

        modes = ["train", "test", "dev"]
        if not os.path.exists(cfg.data_desc_pickle):
            # process the data and extract label_ids and class weights if the model is not restored from .nemo file
            self.data_desc = TokenClassificationDataDesc(
                data_dir=self.data_dir, modes=modes, pad_label=cfg.dataset.pad_label
            )
            if not self.data_desc.data_found:
                raise ValueError(f'No {modes} data found at {self.data_dir}.')
        else:
            self.data_desc = pickle.load(open(cfg.data_desc_pickle, "rb"))
            logging.info(f'Data descriptor restored from {cfg.data_desc_pickle}')

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
        # After this line self._cfg == cfg
        super().__init__(cfg=cfg, trainer=trainer)

        self.bert_model = get_pretrained_lm_model(
            pretrained_model_name=self._cfg.language_model.pretrained_model_name,
            config_file=self._cfg.language_model.bert_config,
            checkpoint_file=self._cfg.language_model.bert_checkpoint,
        )
        self.hidden_size = self.bert_model.hidden_size

        self.classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=self.data_desc.num_classes,
            num_layers=self._cfg.head.num_fc_layers,
            activation=self._cfg.head.activation,
            log_softmax=self._cfg.head.log_softmax,
            dropout=self._cfg.head.fc_dropout,
            use_transformer_init=self._cfg.head.use_transformer_init,
        )

        if self._cfg.dataset.class_balancing == 'weighted_loss':
            # You may need to increase the number of epochs for convergence when using weighted_loss
            self.loss = CrossEntropyLoss(logits_ndim=3, weight=self.data_desc.class_weights)
        else:
            self.loss = CrossEntropyLoss(logits_ndim=3)

        # setup to track metrics
        self.classification_report = ClassificationReport(
            self.data_desc.num_classes, label_ids=self.data_desc.label_ids
        )

    @typecheck()
    def forward(self, input_ids, token_type_ids, attention_mask):
        hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        logits = self.classifier(hidden_states=hidden_states)
        return logits

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, labels = batch
        logits = self(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        loss = self.loss(logits=logits, labels=labels, loss_mask=loss_mask)
        tensorboard_logs = {'train_loss': loss, 'lr': self._optimizer.param_groups[0]['lr']}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, labels = batch
        logits = self(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        val_loss = self.loss(logits=logits, labels=labels, loss_mask=loss_mask)

        subtokens_mask = subtokens_mask > 0.5

        preds = torch.argmax(logits, axis=-1)[subtokens_mask]
        labels = labels[subtokens_mask]
        tp, fp, fn = self.classification_report(preds, labels)

        tensorboard_logs = {'val_loss': val_loss, 'tp': tp, 'fn': fn, 'fp': fp}
        return {'val_loss': val_loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # calculate metrics and log classification report
        tp = torch.sum(torch.stack([x['log']['tp'] for x in outputs]), 0)
        fn = torch.sum(torch.stack([x['log']['fn'] for x in outputs]), 0)
        fp = torch.sum(torch.stack([x['log']['fp'] for x in outputs]), 0)
        precision, recall, f1 = self.classification_report.get_precision_recall_f1(tp, fn, fp, mode='macro')

        tensorboard_logs = {
            'val_loss': avg_loss,
            'precision': precision,
            'f1': f1,
            'recall': recall,
        }
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, prefix='train')

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            self.register_artifact('label_ids.csv', self.data_desc.label_ids_filename)

            if self._cfg.data_desc_pickle is None:
                self._cfg.data_desc_pickle = 'data_desc.p'

            data_desc_file = os.path.join('/tmp', self._cfg.data_desc_pickle)
            pickle.dump(self.data_desc, open(data_desc_file, 'wb'))
            self.register_artifact(config_path='model.data_desc_pickle', src=data_desc_file)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, prefix='dev')

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self.__setup_dataloader_from_config(cfg=test_data_config, prefix='test')

    def _setup_dataloader_from_config(self, cfg: DictConfig, prefix: str):
        text_file = os.path.join(self.data_dir, 'text_' + prefix + '.txt')
        label_file = os.path.join(self.data_dir, 'labels_' + prefix + '.txt')

        if not (os.path.exists(text_file) and os.path.exists(label_file)):
            raise FileNotFoundError(
                f'{text_file} or {label_file} not found. The data should be splitted into 2 files: text.txt and \
                        labels.txt. Each line of the text.txt file contains text sequences, where words are separated with \
                        spaces. The labels.txt file contains corresponding labels for each word in text.txt, the labels are \
                        separated with spaces. Each line of the files should follow the format:  \
                           [WORD] [SPACE] [WORD] [SPACE] [WORD] (for text.txt) and \
                           [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for labels.txt).'
            )
        dataset = BertTokenClassificationDataset(
            text_file=text_file,
            label_file=label_file,
            max_seq_length=self._cfg.dataset.max_seq_length,
            tokenizer=self.tokenizer,
            num_samples=cfg.num_samples,
            pad_label=self.data_desc.pad_label,
            label_ids=self.data_desc.label_ids,
            ignore_extra_tokens=self._cfg.dataset.ignore_extra_tokens,
            ignore_start_end=self._cfg.dataset.ignore_start_end,
            use_cache=self._cfg.dataset.use_cache,
        )

        self.register_artifact('label_ids.csv', self.data_desc.label_ids_filename)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            collate_fn=dataset.collate_fn,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last,
        )

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass
