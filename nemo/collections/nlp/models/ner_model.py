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
from torch.utils.data import DataLoader

from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.common.tokenizers.bert_tokenizer import NemoBertTokenizer
from nemo.collections.nlp.data.token_classification_dataset import BertTokenClassificationDataset
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
        return self.classifier.output_types

    def __init__(
        self,
        num_classes: int,
        pretrained_model_name: str = 'bert-base-cased',
        config_file: Optional[str] = None,
        num_layers: int = 1,
        activation: str = 'relu',
        log_softmax: bool = True,
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
            use_transformer_init: whether to initialize the weights of the classifier head with the same approach used in Transformer
        """
        super().__init__()
        self.bert_model = get_pretrained_lm_model(pretrained_model_name=pretrained_model_name, config_file=config_file)
        self.hidden_size = self.bert_model.config.hidden_size
        self.tokenizer = NemoBertTokenizer(pretrained_model=pretrained_model_name)
        self.classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=num_classes,
            num_layers=num_layers,
            activation=activation,
            log_softmax=log_softmax,
            dropout=dropout,
            use_transformer_init=use_transformer_init,
        )

        self.loss = CrossEntropyLoss()
        # This will be set by setup_training_datai
        self.__train_dl = None
        # This will be set by setup_validation_data
        self.__val_dl = None
        # This will be set by setup_test_data
        self.__test_dl = None
        # This will be set by setup_optimization
        self.__optimizer = None

    @typecheck()
    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
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

        loss = self.loss(logits=logits, labels=labels)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, labels = batch
        logits = self(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        val_loss = self.loss(logits=logits, labels=labels)

        tensorboard_logs = {'val_loss': val_loss}
        return {'val_loss': val_loss, 'log': tensorboard_logs}  # , "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # TODO - add metrics
        # val_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        # tensorboard_logs = {'val_loss': avg_loss, 'val_acc': val_acc}
        return {'val_loss': avg_loss}  # , 'log': tensorboard_logs}

    def setup_training_data(self, data_dir, train_data_layer_params: Optional[Dict]):
        if 'shuffle' not in train_data_layer_params:
            train_data_layer_params['shuffle'] = True
        text_file = os.path.join(data_dir, 'text_train.txt')
        labels_file = os.path.join(data_dir, 'labels_train.txt')
        self.__train_dl = self.__setup_dataloader_ner(text_file, labels_file)

    def setup_validation_data(self, data_dir, val_data_layer_params: Optional[Dict]):
        if 'shuffle' not in val_data_layer_params:
            val_data_layer_params['shuffle'] = False
        text_file = os.path.join(data_dir, 'text_dev.txt')
        labels_file = os.path.join(data_dir, 'labels_dev.txt')
        self.__val_dl = self.__setup_dataloader_ner(text_file, labels_file)

    def setup_test_data(self, test_data_layer_params: Optional[Dict]):
        pass

    def setup_optimization(self, optim_params: Optional[Dict], optimizer='adam'):
        if optimizer == 'adam':
            self.__optimizer = torch.optim.Adam(self.parameters(), lr=optim_params['lr'])
        else:
            raise NotImplementedError()

    def __setup_dataloader_ner(
        self,
        text_file,
        label_file,
        max_seq_length=128,
        pad_label='O',
        label_ids=None,
        num_samples=-1,
        ignore_extra_tokens=False,
        ignore_start_end=False,
        use_cache=False,
        shuffle=False,
        batch_size=64,
        num_workers=0,
    ):

        dataset = BertTokenClassificationDataset(
            text_file=text_file,
            label_file=label_file,
            max_seq_length=max_seq_length,
            tokenizer=self.tokenizer,
            num_samples=num_samples,
            pad_label=pad_label,
            label_ids=label_ids,
            ignore_extra_tokens=ignore_extra_tokens,
            ignore_start_end=ignore_start_end,
            use_cache=use_cache,
        )

        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        )

    def configure_optimizers(self):
        return self.__optimizer

    def train_dataloader(self):
        return self.__train_dl

    def val_dataloader(self):
        return self.__val_dl

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
