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

from typing import Dict, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from nemo.collections.common.losses import SpanningLoss
from nemo.collections.common.tokenizers.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.data.qa_dataset import SquadDataset
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.common_utils import get_pretrained_lm_model
from nemo.core.classes import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import NeuralType
from nemo.utils.decorators import experimental

__all__ = ['QAModel']


@experimental
class QAModel(ModelPT):
    """
    BERT encoder with QA head training.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.classifier.output_types

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.version_2_with_negative = cfg.version_2_with_negative
        self.doc_stride = cfg.doc_stride
        self.max_query_length = cfg.max_query_length
        self.max_seq_length = cfg.max_seq_length

        self.tokenizer = get_tokenizer(
            pretrained_model_name=cfg.language_model.pretrained_model_name, tokenizer_name="nemobert"
        )
        super().__init__(cfg=cfg, trainer=trainer)

        self.bert_model = get_pretrained_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name, config_file=cfg.language_model.bert_config
        )
        self.hidden_size = self.bert_model.config.hidden_size
        self.classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=cfg.token_classifier.num_classes,
            num_layers=cfg.token_classifier.num_layers,
            activation=cfg.token_classifier.activation,
            log_softmax=cfg.token_classifier.log_softmax,
            dropout=cfg.token_classifier.dropout,
            use_transformer_init=cfg.token_classifier.use_transformer_init,
        )

        self.loss = SpanningLoss()

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
        input_ids, input_type_ids, input_mask, unique_ids, start_positions, end_positions = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        loss, _, _ = self.loss(logits=logits, start_positions=start_positions, end_positions=end_positions)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_ids, input_type_ids, input_mask, unique_ids, start_positions, end_positions = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        loss, start_logits, end_logits = self.loss(
            logits=logits, start_positions=start_positions, end_positions=end_positions
        )

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        if outputs:
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

            tensorboard_logs = {'val_loss': avg_loss}
            return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self._setup_dataloader(cfg=test_data_config)

    def _setup_dataloader_from_config(self, cfg: DictConfig):
        dataset = SquadDataset(
            tokenizer=self.tokenizer,
            data_file=cfg.file,
            doc_stride=self.doc_stride,
            max_query_length=self.max_query_length,
            max_seq_length=self.max_seq_length,
            version_2_with_negative=self.version_2_with_negative,
            mode=cfg.mode,
            use_cache=cfg.use_cache,
        )
        dl = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            collate_fn=dataset.collate_fn,
            drop_last=cfg.get('drop_last', False),
            shuffle=cfg.shuffle,
            num_workers=cfg.get('num_workers', 0),
        )
        return dl

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

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass
