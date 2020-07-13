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
from torch.utils.data import DataLoader

from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.common.tokenizers.bert_tokenizer import NemoBertTokenizer
from nemo.collections.nlp.data.text_classification import TextClassificationDataDesc, TextClassificationDataset
from nemo.collections.nlp.modules.common import SequenceClassifier
from nemo.collections.nlp.modules.common.common_utils import get_pretrained_lm_model
from nemo.core.classes import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import NeuralType
from nemo.utils.decorators import experimental

__all__ = ['TextClassificationModel']


@experimental
class TextClassificationModel(ModelPT):
    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.classifier.output_types

    def __init__(
        self,
        data_dir: str,
        pretrained_model_name: str,
        bert_config: str,
        num_output_layers: int,
        fc_dropout: float,
        class_balancing: bool,
    ):
        """
        Initializes the BERTTextClassifier model.
        Args:
            data_dir: the path to the folder containing the data
            pretrained_model_name: name of the BERT model to be used as the encoder
            bert_config: The path to the config file for the BERT encoder, it should be None to use the default configs
            num_output: number of the linear layers of the mlp head on the top of the encoder
            fc_dropout: the dropout used for the mlp head
            class_balancing: enables the weighted class balancing of the loss, may be used for handling unbalanced classes
        """

        # init superclass
        super().__init__()

        data_desc = TextClassificationDataDesc(data_dir=data_dir, modes=["train", "test", "dev"])

        self.bert_model = get_pretrained_lm_model(pretrained_model_name=pretrained_model_name, config_file=bert_config)
        self.hidden_size = self.bert_model.config.hidden_size
        self.tokenizer = NemoBertTokenizer(pretrained_model=pretrained_model_name)
        self.classifier = SequenceClassifier(
            hidden_size=self.hidden_size,
            num_classes=data_desc.num_labels,
            dropout=fc_dropout,
            num_layers=num_output_layers,
            log_softmax=False,
        )

        if class_balancing == 'weighted_loss':
            # You may need to increase the number of epochs for convergence when using weighted_loss
            self.loss = CrossEntropyLoss(weight=data_desc.class_weights)
        else:
            self.loss = CrossEntropyLoss()

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
        input_ids, input_type_ids, input_mask, labels = batch
        logits = self(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        # TODO replace with loss module
        train_loss = self.loss(logits=logits, labels=labels)

        tensorboard_logs = {'train_loss': train_loss}
        return {'loss': train_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, labels = batch
        logits = self(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        val_loss = self.loss(logits=logits, labels=labels)

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

    def setup_training_data(self, file_path, dataloader_params={}):
        self._train_dl = self.__setup_dataloader(input_file=file_path, dataloader_params=dataloader_params)

    def setup_validation_data(self, file_path, dataloader_params={}):
        self._validation_dl = self.__setup_dataloader(input_file=file_path, dataloader_params=dataloader_params)

    def setup_test_data(self, file_path, dataloader_params={}):
        self._test_dl = self.__setup_dataloader(input_file=file_path, dataloader_params=dataloader_params)

    def __setup_dataloader(
        self, input_file, dataloader_params={},
    ):
        dataset = TextClassificationDataset(
            input_file=input_file,
            tokenizer=self.tokenizer,
            max_seq_length=dataloader_params.get("max_seq_length", 512),
            num_samples=dataloader_params.get("num_samples", -1),
            shuffle=dataloader_params.get("shuffle", False),
            use_cache=dataloader_params.get("use_cache", False),
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=dataloader_params.get("batch_size", 64),
            shuffle=dataloader_params.get("shuffle", False),
            num_workers=dataloader_params.get("num_workers", 0),
            pin_memory=dataloader_params.get("pin_memory", False),
        )

    @classmethod
    def save_to(self, save_path: str):
        """
        Saves the module to the specified path.
        Args:
            :param save_path: Path to where to save the module.
        """
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        """
        Restores the module from the specified path.
        Args:
            :param restore_path: Path to restore the module from.
        """
        pass

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass

    @classmethod
    def from_pretrained(cls, name: str):
        pass

    @classmethod
    def export(self, **kwargs):
        pass
