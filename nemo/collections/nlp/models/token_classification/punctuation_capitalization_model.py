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
from torch.utils.data import DataLoader

from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss
from nemo.collections.common.tokenizers.bert_tokenizer import NemoBertTokenizer
from nemo.collections.nlp.data.punctuation_capitalization_dataset import BertPunctuationCapitalizationDataset
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.common_utils import get_pretrained_lm_model
from nemo.core.classes import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import LogitsType, NeuralType
from nemo.utils.decorators import experimental

__all__ = ['PunctuationCapitalizationModel']


@experimental
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

    def __init__(
        self,
        punct_num_classes: int = 4,
        capit_num_classes: int = 2,
        none_label: str = 'O',
        pretrained_model_name: str = 'bert-base-cased',
        config_file: str = None,
        activation: str = 'relu',
        log_softmax: bool = True,
        dropout: float = 0.0,
        use_transformer_init: bool = True,
        tokenizer_type: str = 'nemobert',
        punct_num_fc_layers: int = 1,
        capit_num_fc_layers: int = 1,
    ):
        """
        Initializes BERT Punctuation and Capitalization model.
        Args:
            punct_num_classes: number of classes for pucntuation task
            capit_num_classes: number of classes for capitalization task
            none_label: none label used for padding
            pretrained_model_name: pretrained language model name, to see the complete list use
            config_file: model config file
            activation: activation to usee between fully connected layers in the MLP
            log_softmax: whether to apply softmax to the output
            dropout: dropout to apply to the input hidden states
            use_transformer_init: whether to initialize the weights of the classifier head with the same approach used in Transformer
            tokenizer_type: tokenizer type: nemobert or sentencepiece
            punct_num_fc_layers: number of fully connected layers in the multilayer perceptron (MLP)
            capit_num_fc_layers: number of fully connected layers in the multilayer perceptron (MLP)
        """
        super().__init__()
        self.none_label = none_label
        self.bert_model = get_pretrained_lm_model(pretrained_model_name=pretrained_model_name, config_file=config_file)
        self.hidden_size = self.bert_model.config.hidden_size

        # TODO add support for sentence_piece tokenizer
        if tokenizer_type == 'nemobert':
            self.tokenizer = NemoBertTokenizer(pretrained_model=pretrained_model_name)
        else:
            raise NotImplementedError()

        # TODO refactor with data_desc
        self.punct_classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=punct_num_classes,
            activation=activation,
            log_softmax=log_softmax,
            dropout=dropout,
            num_layers=punct_num_fc_layers,
            use_transformer_init=use_transformer_init,
        )

        self.capit_classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=capit_num_classes,
            activation=activation,
            log_softmax=log_softmax,
            dropout=dropout,
            num_layers=capit_num_fc_layers,
            use_transformer_init=use_transformer_init,
        )

        self.loss = CrossEntropyLoss()
        self.agg_loss = AggregatorLoss(num_inputs=2)

        # TODO fix with config
        self.max_seq_length = 128
        self.pad_label = 'O'
        self.punct_label_ids = None
        self.capit_label_ids = None
        self.num_samples = -1
        self.ignore_extra_tokens = False
        self.ignore_start_end = False
        self.overwrite_processed_files = False
        self.shuffle = False
        self.batch_size = 64
        self.num_workers = 0

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
        punct_logits, capit_logits = self(
            input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask
        )

        punct_loss = self.loss(logits=punct_logits, labels=punct_labels, loss_mask=loss_mask)
        capit_loss = self.loss(logits=capit_logits, labels=capit_labels, loss_mask=loss_mask)
        loss = self.agg_loss(loss_1=punct_loss, loss_2=capit_loss)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
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

        tensorboard_logs = {'val_loss': val_loss}
        return {'val_loss': val_loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    def setup_training_data(self, data_dir, train_data_layer_params: Optional[Dict]):
        if 'shuffle' not in train_data_layer_params:
            train_data_layer_params['shuffle'] = True
        text_file = os.path.join(data_dir, 'text_train.txt')
        labels_file = os.path.join(data_dir, 'labels_train.txt')
        self._train_dl = self.__setup_dataloader(text_file, labels_file)

    def setup_validation_data(self, data_dir, val_data_layer_params: Optional[Dict]):
        if 'shuffle' not in val_data_layer_params:
            val_data_layer_params['shuffle'] = False
        text_file = os.path.join(data_dir, 'text_dev.txt')
        labels_file = os.path.join(data_dir, 'labels_dev.txt')
        self._validation_dl = self.__setup_dataloader(text_file, labels_file)

    def setup_test_data(self, test_data_layer_params: Optional[Dict]):
        pass

    def __setup_dataloader(self, text_file: str, label_file: str):

        dataset = BertPunctuationCapitalizationDataset(
            tokenizer=self.tokenizer,
            text_file=text_file,
            label_file=label_file,
            pad_label=self.pad_label,
            punct_label_ids=self.punct_label_ids,
            capit_label_ids=self.capit_label_ids,
            max_seq_length=self.max_seq_length,
            ignore_extra_tokens=self.ignore_extra_tokens,
            ignore_start_end=self.ignore_start_end,
            overwrite_processed_files=self.overwrite_processed_files,
            num_samples=self.num_samples,
        )

        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers,
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
