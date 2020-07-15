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
from dataclasses import dataclass
from typing import Dict, Optional, Union

import hydra
import torch
from omegaconf import MISSING, DictConfig

from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss
from nemo.collections.common.tokenizers.bert_tokenizer import NemoBertTokenizer
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
from nemo.collections.nlp.data.punctuation_capitalization_dataset import BertPunctuationCapitalizationDataset
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.common_utils import get_pretrained_lm_model
from nemo.core.classes import typecheck
from nemo.core.classes.modelPT import ModelPT, ModelPTConfig
from nemo.core.config.optimizers import AdamParams, OptimizerArgs
from nemo.core.config.schedulers import SchedulerArgs, WarmupAnnealingParams
from nemo.core.neural_types import LogitsType, NeuralType
from nemo.utils.decorators import experimental

__all__ = ['PunctuationCapitalizationModel', 'PunctuationCapitalizationModelConfig']


@dataclass
class PunctuationCapitalizationOptimizationConfig:
    name: str = "adam"
    lr: float = 2e-5
    args: OptimizerArgs = OptimizerArgs(params=AdamParams(weight_decay=0.01))
    sched: SchedulerArgs = SchedulerArgs(params=WarmupAnnealingParams(warmup_ratio=0.1))


@dataclass
class PunctuationCapitalizationDataConfig:
    max_seq_length: int = 128
    pad_label: str = 'O'
    punct_label_ids: dict = None
    capit_label_ids: dict = None
    num_samples: int = -1
    ignore_extra_tokens: bool = False
    ignore_start_end: bool = False
    overwrite_processed_files: bool = False
    shuffle: bool = True
    batch_size: int = 64
    num_workers: int = 2
    pin_memory: bool = True
    drop_last: bool = False


@dataclass
class SentencePieceTokenizerConfig:
    name: str = 'sentencepiece'
    model: str = MISSING


@dataclass
class NemoBertTokenizerConfig:
    name: str = 'nemobert'


@dataclass
class LanguageModelConfig:
    pretrained_model_name: Optional[str] = 'bert-base-uncased'
    bert_config: Optional[DictConfig] = None
    tokenizer: Optional[Union[SentencePieceTokenizerConfig, NemoBertTokenizerConfig]] = NemoBertTokenizerConfig()


@dataclass
class PunctuationHeadConfig:
    punct_num_classes: int = 4
    punct_num_fc_layers: int = 1
    fc_dropout: float = 0.1
    activation: str = 'relu'
    log_softmax: bool = True
    use_transformer_init: bool = True


@dataclass
class CapitalizationHeadConfig:
    capit_num_classes: int = 2
    capit_num_fc_layers: int = 1
    fc_dropout: float = 0.1
    activation: str = 'relu'
    log_softmax: bool = True
    use_transformer_init: bool = True


@dataclass
class PunctuationCapitalizationModelConfig(ModelPTConfig):
    language_model: LanguageModelConfig = MISSING
    punct_head: PunctuationHeadConfig = MISSING
    capit_head: CapitalizationHeadConfig = MISSING
    class_balancing: Optional[str] = None
    data_dir: str = MISSING
    tokenizer_type: str = 'nemobert'
    train_ds: PunctuationCapitalizationDataConfig = MISSING
    validation_ds: PunctuationCapitalizationDataConfig = MISSING
    test_ds: Optional[PunctuationCapitalizationDataConfig] = None
    optim: PunctuationCapitalizationOptimizationConfig = PunctuationCapitalizationOptimizationConfig()


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

    def __init__(self, cfg: PunctuationCapitalizationModelConfig):
        """
        Initializes BERT Punctuation and Capitalization model.
        Args:
            # TODO add to hydra:
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
        self.data_dir = cfg.data_dir
        # TODO add support for sentence_piece tokenizer
        if PunctuationCapitalizationModelConfig.tokenizer_type == 'nemobert':
            self.tokenizer = NemoBertTokenizer(pretrained_model=cfg.language_model.pretrained_model_name)
        else:
            raise NotImplementedError()

        super().__init__(cfg=cfg)

        self.bert_model = get_pretrained_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name, config_file=cfg.language_model.bert_config
        )
        self.hidden_size = self.bert_model.config.hidden_size

        # TODO refactor with data_desc
        self.punct_classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=PunctuationHeadConfig.punct_num_classes,
            activation=PunctuationHeadConfig.activation,
            log_softmax=PunctuationHeadConfig.log_softmax,
            dropout=PunctuationHeadConfig.fc_dropout,
            num_layers=PunctuationHeadConfig.punct_num_fc_layers,
            use_transformer_init=PunctuationHeadConfig.use_transformer_init,
        )

        self.capit_classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=CapitalizationHeadConfig.capit_num_classes,
            activation=CapitalizationHeadConfig.activation,
            log_softmax=CapitalizationHeadConfig.log_softmax,
            dropout=CapitalizationHeadConfig.fc_dropout,
            num_layers=CapitalizationHeadConfig.capit_num_fc_layers,
            use_transformer_init=CapitalizationHeadConfig.use_transformer_init,
        )

        self.loss = CrossEntropyLoss()
        self.agg_loss = AggregatorLoss(num_inputs=2)

        # Optimizer setup needs to happen after all model weights are ready
        self.setup_optimization(cfg.optim)

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
        tensorboard_logs = {'train_loss': loss, 'lr': self._optimizer.param_groups[0]['lr']}
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

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config)

    def setup_validation_data(self, val_data_config: Optional[Dict]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config)

    def setup_test_data(self, test_data_config: Optional[Dict]):
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config)

    def _setup_dataloader_from_config(self, cfg: PunctuationCapitalizationDataConfig):
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
        dataset = BertPunctuationCapitalizationDataset(
            tokenizer=self.tokenizer,
            text_file=text_file,
            label_file=label_file,
            pad_label=cfg.pad_label,
            punct_label_ids=cfg.punct_label_ids,
            capit_label_ids=cfg.capit_label_ids,
            max_seq_length=cfg.max_seq_length,
            ignore_extra_tokens=cfg.ignore_extra_tokens,
            ignore_start_end=cfg.ignore_start_end,
            overwrite_processed_files=cfg.overwrite_processed_files,
            num_samples=cfg.num_samples,
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
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
