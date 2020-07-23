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
from nemo.utils.decorators import experimental

__all__ = ['NERModel']


@experimental
class NERModel(ModelPT):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.output_types

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        """
        Initializes BERT Named Entity Recognition model.
        """

        self.data_desc = TokenClassificationDataDesc(
            data_dir=cfg.data_dir, modes=["train", "test", "dev"], pad_label=cfg.pad_label
        )
        self.data_dir = cfg.data_dir
        self.model_cfg = cfg

        self.tokenizer = get_tokenizer(
            tokenizer_name=cfg.language_model.tokenizer,
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            vocab_file=cfg.language_model.vocab_file,
            tokenizer_model=cfg.language_model.tokenizer_model,
            do_lower_case=cfg.language_model.do_lower_case,
        )

        super().__init__(cfg=cfg, trainer=trainer)

        self.bert_model = get_pretrained_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            config_file=cfg.language_model.bert_config,
            checkpoint_file=cfg.language_model.bert_checkpoint,
        )
        self.hidden_size = self.bert_model.config.hidden_size

        self.classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=self.data_desc.num_classes,
            num_layers=cfg.head.num_fc_layers,
            activation=cfg.head.activation,
            log_softmax=cfg.head.log_softmax,
            dropout=cfg.head.fc_dropout,
            use_transformer_init=cfg.head.use_transformer_init,
        )

        if cfg.class_balancing == 'weighted_loss':
            # You may need to increase the number of epochs for convergence when using weighted_loss
            self.loss = CrossEntropyLoss(weight=self.data_desc.class_weights)
        else:
            self.loss = CrossEntropyLoss()

        # setup to track metrics
        self.classification_report = ClassificationReport(
            self.data_desc.num_classes, label_ids=self.data_desc.label_ids
        )
        # Optimizer setup needs to happen after all model weights are ready
        self.setup_optimization(cfg.optim)

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
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        loss = self.loss(logits=logits, labels=labels, loss_mask=loss_mask)
        tensorboard_logs = {'train_loss': loss, 'lr': self._optimizer.param_groups[0]['lr']}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, labels = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
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
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self.__setup_dataloader(cfg=test_data_config)

    def _setup_dataloader_from_config(self, cfg: DictConfig):
        text_file = os.path.join(self.data_dir, 'text_' + cfg.prefix + '.txt')
        label_file = os.path.join(self.data_dir, 'labels_' + cfg.prefix + '.txt')

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
            max_seq_length=self.model_cfg.max_seq_length,
            tokenizer=self.tokenizer,
            num_samples=cfg.num_samples,
            pad_label=self.data_desc.pad_label,
            label_ids=self.data_desc.label_ids,
            ignore_extra_tokens=self.model_cfg.ignore_extra_tokens,
            ignore_start_end=self.model_cfg.ignore_start_end,
            use_cache=self.model_cfg.use_cache,
        )

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

    @classmethod
    def from_pretrained(cls, name: str):
        pass

    def export(self, **kwargs):
        pass

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass
