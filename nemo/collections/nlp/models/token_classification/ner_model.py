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
from pytorch_lightning.metrics.functional import f1_score as f1_pl
from pytorch_lightning.metrics.sklearns import F1 as f1_pl_sklearn
from sklearn.metrics import f1_score as f1_sklearn
from torch.utils.data import DataLoader

from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.common.tokenizers.bert_tokenizer import NemoBertTokenizer
from nemo.collections.nlp.data.token_classification.token_classification_dataset import BertTokenClassificationDataset
from nemo.collections.nlp.metrics.metrics_utils import get_classification_report
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.common_utils import get_pretrained_lm_model
from nemo.core.classes import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import NeuralType
from nemo.utils import logging
from nemo.utils.decorators import experimental
from nemo.collections.nlp.data.token_classification.token_classification_descriptor import TokenClassificationDataDesc

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
        data_dir: str,
        pretrained_model_name: str = 'bert-base-cased',
        config_file: Optional[str] = None,
        num_layers: int = 1,
        activation: str = 'relu',
        log_softmax: bool = True,
        dropout: float = 0.0,
        class_balancing: bool = False,
        use_transformer_init: bool = True,
    ):
        """
        Initializes BERT Named Entity Recognition model.
        Args:
            data_dir: the path to the folder containing the data
            num_classes: number of classes
            pretrained_model_name: pretrained language model name, to see the complete list use
            config_file: model config file
            num_layers: number of fully connected layers in the multilayer perceptron (MLP)
            activation: activation to usee between fully connected layers in the MLP
            log_softmax: whether to apply softmax to the output
            dropout: dropout to apply to the input hidden states
            class_balancing: enables the weighted class balancing of the loss, may be used for handling unbalanced classes
            use_transformer_init: whether to initialize the weights of the classifier head with the same approach used in Transformer
        """
        super().__init__()

        self.data_desc = TokenClassificationDataDesc(data_dir=data_dir, modes=["train", "test", "dev"], pad_label='O')
        self.bert_model = get_pretrained_lm_model(pretrained_model_name=pretrained_model_name, config_file=config_file)
        self.hidden_size = self.bert_model.config.hidden_size
        self.tokenizer = NemoBertTokenizer(pretrained_model=pretrained_model_name)
        self.classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=self.data_desc.num_classes,
            num_layers=num_layers,
            activation=activation,
            log_softmax=log_softmax,
            dropout=dropout,
            use_transformer_init=use_transformer_init,
        )

        if class_balancing == 'weighted_loss':
            # You may need to increase the number of epochs for convergence when using weighted_loss
            self.loss = CrossEntropyLoss(weight=self.data_desc.class_weights)
        else:
            self.loss = CrossEntropyLoss()

        self.overwrite_processed_files = False
        self.max_seq_length = 128
        self.num_samples = -1
        self.ignore_extra_tokens = False
        self.ignore_start_end = False
        self.use_cache = False
        self.shuffle = False
        self.batch_size = 64
        self.num_workers = 0
        self.shuffle = False
        self.num_workers = 0
        self.batch_size = 8

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
        tensorboard_logs = {'train_loss': loss}
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
        preds = torch.argmax(logits, axis=-1)
        labels = labels
        tensorboard_logs = {'val_loss': val_loss, 'preds': preds, 'labels': labels, 'subtokens_mask': subtokens_mask}
        return {'val_loss': val_loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        subtokens_mask = torch.stack([x['log']['subtokens_mask'] for x in outputs])
        preds = torch.stack([x['log']['preds'] for x in outputs])[subtokens_mask]
        labels = torch.stack([x['log']['labels'] for x in outputs])[subtokens_mask]

        val_f1_pl = f1_pl(pred=preds, target=labels) * 100  # , 2)
        logging.info(f'F1 pl: {val_f1_pl}')

        f1_pl_sklearn_metric = f1_pl_sklearn(average='macro')
        # val_f1_pl_sklearn = f1_pl_sklearn_metric(preds, labels)[0] * 100
        # logging.info(f'F1 pl sklearn: {val_f1_pl_sklearn}')

        # val_f1_sklearn = round(f1_sklearn(labels, preds, average='macro') * 100, 2)
        # logging.info(f'F1 sklearn: {val_f1_sklearn}')

        # class_report = get_classification_report(labels=labels, preds=preds)
        # TO DO remove .numpy()
        # class_report = get_classification_report(labels=labels.numpy(), preds=preds.numpy(), label_ids=self.data_desc.label_ids)
        # logging.info(class_report)
        # val_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        tensorboard_logs = {'val_loss': avg_loss, 'val_f1': val_f1_pl}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def setup_training_data(self, data_dir, train_data_layer_config: Optional[Dict]):
        if 'shuffle' not in train_data_layer_config:
            train_data_layer_config['shuffle'] = True
        text_file = os.path.join(data_dir, 'text_train.txt')
        labels_file = os.path.join(data_dir, 'labels_train.txt')
        self._train_dl = self.__setup_dataloader_ner(text_file, labels_file)

    def setup_validation_data(self, data_dir, val_data_layer_config: Optional[Dict]):
        if 'shuffle' not in val_data_layer_config:
            val_data_layer_config['shuffle'] = False
        text_file = os.path.join(data_dir, 'text_dev.txt')
        labels_file = os.path.join(data_dir, 'labels_dev.txt')
        self._validation_dl = self.__setup_dataloader_ner(text_file, labels_file)

    def setup_test_data(self, test_data_layer_params: Optional[Dict]):
        pass

    def __setup_dataloader_ner(self, text_file, label_file):

        dataset = BertTokenClassificationDataset(
            text_file=text_file,
            label_file=label_file,
            max_seq_length=self.max_seq_length,
            tokenizer=self.tokenizer,
            num_samples=self.num_samples,
            pad_label=self.data_desc.pad_label,
            label_ids=self.data_desc.label_ids,
            ignore_extra_tokens=self.ignore_extra_tokens,
            ignore_start_end=self.ignore_start_end,
            overwrite_processed_files=self.overwrite_processed_files,
        )

        return torch.utils.data.DataLoader(
                dataset=dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers
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
