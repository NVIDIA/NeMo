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
import torch.nn as nn
from torch.utils.data import DataLoader

from nemo.collections.common.tokenizers.bert_tokenizer import NemoBertTokenizer
from nemo.collections.nlp.data.punctuation_capitalization_dataset import BertPunctuationCapitalizationDataset
from nemo.collections.nlp.modules.common import TokenClassifier

from nemo.collections.nlp.modules.common.huggingface.bert import BertEncoder
from nemo.core.classes import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import NeuralType, LogitsType
from nemo.utils.decorators import experimental

__all__ = ['PunctuationCapitalizationModel']


@experimental
class PunctuationCapitalizationModel(ModelPT):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"punct_logits": NeuralType(('B', 'T', 'C'), LogitsType()), "capit_logits": NeuralType(('B', 'T', 'C'), LogitsType())}

    def __init__(
        self,
        punct_num_classes=4,
        capit_num_classes=2,
        none_label='O',
        pretrained_model_name='bert-base-cased',
        activation='relu',
        log_softmax=True,
        dropout=0.0,
        use_transformer_pretrained=True,
        tokenizer_type='nemobert',
        punct_num_fc_layers=1,
        capit_num_fc_layers=1

    ):
        # init superclass
        super().__init__()
        self.none_label = none_label
        self.bert_model = BertEncoder.from_pretrained(pretrained_model_name)
        self.hidden_size = self.bert_model.config.hidden_size

        # TODO add support for sentence_piece tokenizer
        if tokenizer_type == 'nemobert':
            self.tokenizer = NemoBertTokenizer(pretrained_model=pretrained_model_name)
        else:
            raise NotImplementedError()

        self.punct_classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=punct_num_classes,
            activation=activation,
            log_softmax=log_softmax,
            dropout=dropout,
            num_layers=punct_num_fc_layers,
            use_transformer_pretrained=use_transformer_pretrained,
        )

        self.capit_classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=capit_num_classes,
            activation=activation,
            log_softmax=log_softmax,
            dropout=dropout,
            num_layers=capit_num_fc_layers,
            use_transformer_pretrained=use_transformer_pretrained,
        )

        self.loss = nn.CrossEntropyLoss()
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
        punct_logits = self.punct_classifier(hidden_states=hidden_states)
        capit_logits = self.capit_classifier(hidden_states=hidden_states)
        return punct_logits, capit_logits

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, punct_labels, capit_labels = batch
        punct_logits, capit_logits = self(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        # TODO replace with loss module
        punct_logits_flatten = torch.flatten(punct_logits, start_dim=0, end_dim=-2)
        punct_labels_flatten = torch.flatten(punct_labels, start_dim=0, end_dim=-1)
        punct_loss = self.loss(punct_logits_flatten, punct_labels_flatten)

        capit_logits_flatten = torch.flatten(capit_logits, start_dim=0, end_dim=-2)
        capit_labels_flatten = torch.flatten(capit_labels, start_dim=0, end_dim=-1)
        capit_loss = self.loss(capit_logits_flatten, capit_labels_flatten)

        loss = punct_loss + capit_loss
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, loss_mask, subtokens_mask, punct_labels, capit_labels = batch
        punct_logits, capit_logits = self(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        # TODO replace with loss module
        punct_logits_flatten = torch.flatten(punct_logits, start_dim=0, end_dim=-2)
        punct_labels_flatten = torch.flatten(punct_labels, start_dim=0, end_dim=-1)
        punct_loss = self.loss(punct_logits_flatten, punct_labels_flatten)

        capit_logits_flatten = torch.flatten(capit_logits, start_dim=0, end_dim=-2)
        capit_labels_flatten = torch.flatten(capit_labels, start_dim=0, end_dim=-1)
        capit_loss = self.loss(capit_logits_flatten, capit_labels_flatten)

        val_loss = punct_loss + capit_loss

        tensorboard_logs = {'val_loss': val_loss}
        # TODO - add eval - callback?
        # labels_hat = torch.argmax(y_hat, dim=1)
        # n_correct_pred = torch.sum(y == labels_hat).item()
        return {'val_loss': val_loss, 'log': tensorboard_logs}  # , "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # val_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        # tensorboard_logs = {'val_loss': avg_loss, 'val_acc': val_acc}
        return {'val_loss': avg_loss}  # , 'log': tensorboard_logs}

    def setup_training_data(self, data_dir, train_data_layer_params: Optional[Dict]):
        if 'shuffle' not in train_data_layer_params:
            train_data_layer_params['shuffle'] = True
        text_file = os.path.join(data_dir, 'text_train.txt')
        labels_file = os.path.join(data_dir, 'labels_train.txt')
        self.__train_dl = self.__setup_dataloader(text_file, labels_file)

    def setup_validation_data(self, data_dir, val_data_layer_params: Optional[Dict]):
        if 'shuffle' not in val_data_layer_params:
            val_data_layer_params['shuffle'] = False
        text_file = os.path.join(data_dir, 'text_dev.txt')
        labels_file = os.path.join(data_dir, 'labels_dev.txt')
        self.__val_dl = self.__setup_dataloader(text_file, labels_file)

    def setup_test_data(self, test_data_layer_params: Optional[Dict]):
        pass

    def setup_optimization(self, optim_params: Optional[Dict], optimizer='adam'):
        if optimizer == 'adam':
            self.__optimizer = torch.optim.Adam(self.parameters(), lr=optim_params['lr'])
        else:
            raise NotImplementedError()

    def __setup_dataloader(
        self,
        text_file,
        label_file,
        max_seq_length=128,
        pad_label='O',
        punct_label_ids=None,
        capit_label_ids=None,
        num_samples=-1,
        ignore_extra_tokens=False,
        ignore_start_end=False,
        overwrite_processed_files=False,
        shuffle=False,
        batch_size=64,
        num_workers=0,
    ):

        dataset = BertPunctuationCapitalizationDataset(
            tokenizer=self.tokenizer,
            text_file=text_file,
            label_file=label_file,
            pad_label=pad_label,
            punct_label_ids=punct_label_ids,
            capit_label_ids=capit_label_ids,
            max_seq_length=max_seq_length,
            ignore_extra_tokens=ignore_extra_tokens,
            ignore_start_end=ignore_start_end,
            overwrite_processed_files=overwrite_processed_files,
            num_samples=num_samples
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
