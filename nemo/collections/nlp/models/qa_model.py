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
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nemo.collections.common.losses import SpanningLoss
from nemo.collections.common.tokenizers.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.data.qa_dataset import SquadDataset
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.common_utils import get_pretrained_lm_model
from nemo.collections.nlp.modules.common.huggingface.bert import BertEncoder
from nemo.core.classes import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import NeuralType
from nemo.core.optim import prepare_lr_scheduler
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

    def __init__(
        self,
        num_classes: int = 2,
        pretrained_model_name: str = 'bert-base-cased',
        config_file: Optional[str] = None,
        num_layers: int = 1,
        activation: str = 'relu',
        log_softmax: bool = False,
        dropout: float = 0.0,
        use_transformer_init: bool = True,
    ):
        """
        Args:
            num_classes: number of classes
            pretrained_model_name: pretrained language model name, to see the complete list use
            config_file: model config file
            num_layers: number of fully connected layers in the multilayer perceptron (MLP)
            activation: activation to usee between fully connected layers in the MLP
            log_softmax: whether to apply softmax to the output
            dropout: dropout to apply to the input hidden states
            use_transformer_init: whether to use pre-trained transformer weights for weights initialization
        """
        # init superclass
        super().__init__()
        self.bert_model = get_pretrained_lm_model(pretrained_model_name=pretrained_model_name, config_file=config_file)
        self.hidden_size = self.bert_model.config.hidden_size
        self.tokenizer = get_tokenizer(pretrained_model_name=pretrained_model_name, tokenizer_name="nemobert")
        self.classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=num_classes,
            num_layers=num_layers,
            activation=activation,
            log_softmax=log_softmax,
            dropout=dropout,
            use_transformer_init=use_transformer_init,
        )

        self.loss = SpanningLoss()
        # This will be set by setup_training_datai
        self.__train_dl = None
        # This will be set by setup_validation_data
        self.__val_dl = None
        # This will be set by setup_test_data
        self.__test_dl = None
        # This will be set by setup_optimization
        self.__optimizer = None
        # This will be set by setup_optimization
        self.__scheduler = None

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
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def setup_training_data(self, train_data_layer_config: Optional[Dict]):
        if 'shuffle' not in train_data_layer_config:
            train_data_layer_config['shuffle'] = True
            train_data_layer_config['mode'] = 'train'
        self.__train_dl = self.__setup_dataloader(train_data_layer_config)

    def setup_validation_data(self, val_data_layer_config: Optional[Dict]):
        if 'shuffle' not in val_data_layer_config:
            val_data_layer_config['shuffle'] = False
            val_data_layer_config['mode'] = 'eval'
        self.__val_dl = self.__setup_dataloader(val_data_layer_config)

    def setup_test_data(self, test_data_layer_params: Optional[Dict]):
        pass

    def setup_optimization(self, optim_config: Optional[Dict] = None) -> torch.optim.Optimizer:
        self.__optimizer = super().setup_optimization(optim_config)
        self.__scheduler = prepare_lr_scheduler(
            optimizer=self.__optimizer, scheduler_config=optim_config, train_dataloader=self.__train_dl
        )

    def __setup_dataloader(self, data_layer_params):
        dataset = SquadDataset(
            tokenizer=self.tokenizer,
            data_file=data_layer_params['data_file'],
            doc_stride=data_layer_params['doc_stride'],
            max_query_length=data_layer_params['max_query_length'],
            max_seq_length=data_layer_params['max_seq_length'],
            version_2_with_negative=data_layer_params['version_2_with_negative'],
            mode=data_layer_params['mode'],
            use_cache=data_layer_params['use_cache'],
        )
        dl = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=data_layer_params['batch_size'],
            collate_fn=dataset.collate_fn,
            drop_last=data_layer_params.get('drop_last', False),
            shuffle=data_layer_params['shuffle'],
            num_workers=data_layer_params.get('num_workers', 0),
        )
        return dl

    def configure_optimizers(self):
        if self.__scheduler is None:
            return self.__optimizer
        else:
            return [self.__optimizer], [self.__scheduler]

    def train_dataloader(self):
        return self.__train_dl

    def val_dataloader(self):
        return self.__val_dl

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
