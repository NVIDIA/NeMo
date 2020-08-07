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

import json
import os
from typing import Dict, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss, SmoothedCrossEntropyLoss
from nemo.collections.common.tokenizers.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.data.language_modeling.lm_bert_dataset import (
    BertPretrainingDataset,
    BertPretrainingPreprocessedDataloader,
)
from nemo.collections.nlp.metrics.perplexity import Perplexity
from nemo.collections.nlp.modules.common import BertPretrainingTokenClassifier, SequenceClassifier
from nemo.collections.nlp.modules.common.common_utils import get_pretrained_lm_model
from nemo.core.classes import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import NeuralType
from nemo.utils import logging

__all__ = ['BERTLMModel']


class BERTLMModel(ModelPT):
    """
    BERT language model pretraining.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        output_types_dict = {'mlm_logits': self.mlm_classifier.output_types['logits']}
        if not self.only_mlm_loss:
            output_types_dict['nsp_logits'] = self.nsp_classifier.output_types['logits']
        return output_types_dict

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        if cfg.language_model.bert_config_file is not None:
            logging.info(
                (
                    f"HuggingFace BERT config file found. "
                    f"LM will be instantiated from: {cfg.language_model.bert_config_file}"
                )
            )
            self.vocab_size = json.load(open(cfg.language_model.bert_config_file))['vocab_size']
        elif cfg.language_model.bert_config.vocab_size is not None:
            self.vocab_size = cfg.language_model.bert_config.vocab_size
        else:
            self.vocab_size = None

        if cfg.tokenizer is not None:
            cfg.tokenizer.vocab_size = self.vocab_size
            self._setup_tokenizer(cfg.tokenizer)
        else:
            self.tokenizer = None

        super().__init__(cfg=cfg, trainer=trainer)

        if cfg.language_model.bert_config_file is not None:
            self.bert_model = get_pretrained_lm_model(
                pretrained_model_name=cfg.language_model.pretrained_model_name,
                config_file=cfg.language_model.bert_config_file,
                checkpoint_file=cfg.language_model.bert_checkpoint,
            )
        else:
            self.bert_model = get_pretrained_lm_model(
                pretrained_model_name=cfg.language_model.pretrained_model_name,
                config_dict=OmegaConf.to_container(cfg.language_model.bert_config),
                checkpoint_file=cfg.language_model.bert_checkpoint,
            )

        self.hidden_size = self.bert_model.config.hidden_size
        self.vocab_size = self.bert_model.config.vocab_size
        self.only_mlm_loss = cfg.only_mlm_loss

        self.mlm_classifier = BertPretrainingTokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=self.vocab_size,
            num_layers=cfg.num_tok_classification_layers,
            activation='gelu',
            log_softmax=True,
            use_transformer_init=True,
        )

        self.mlm_loss = SmoothedCrossEntropyLoss()

        if not self.only_mlm_loss:
            self.nsp_classifier = SequenceClassifier(
                hidden_size=self.hidden_size,
                num_classes=2,
                num_layers=cfg.num_seq_classification_layers,
                log_softmax=False,
                activation='tanh',
                use_transformer_init=True,
            )

            self.nsp_loss = CrossEntropyLoss()
            self.agg_loss = AggregatorLoss(num_inputs=2)

        # # tie weights of MLM softmax layer and embedding layer of the encoder
        if (
            self.mlm_classifier.mlp.last_linear_layer.weight.shape
            != self.bert_model.embeddings.word_embeddings.weight.shape
        ):
            raise ValueError("Final classification layer does not match embedding layer.")
        self.mlm_classifier.mlp.last_linear_layer.weight = self.bert_model.embeddings.word_embeddings.weight
        # create extra bias

        # setup to track metrics
        self.perplexity_metric = Perplexity()

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
        mlm_logits = self.mlm_classifier(hidden_states=hidden_states)
        if self.only_mlm_loss:
            return (mlm_logits,)

        nsp_logits = self.nsp_classifier(hidden_states=hidden_states)
        return mlm_logits, nsp_logits

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        input_ids, input_type_ids, input_mask, output_ids, output_mask, labels = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        mlm_loss = self.mlm_loss(logits=logits[0], labels=output_ids, output_mask=output_mask)

        if self.only_mlm_loss:
            loss = mlm_loss
        else:
            nsp_loss = self.nsp_loss(logits=logits[1], labels=labels)

            loss = self.agg_loss(loss_1=mlm_loss, loss_2=nsp_loss)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, output_ids, output_mask, labels = batch
        logits = self.forward(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)

        mlm_loss = self.mlm_loss(logits=logits[0], labels=output_ids, output_mask=output_mask)

        if self.only_mlm_loss:
            loss = mlm_loss
        else:
            nsp_loss = self.nsp_loss(logits=logits[1], labels=labels)

            loss = self.agg_loss(loss_1=mlm_loss, loss_2=nsp_loss)
        perplexity = self.perplexity_metric(mlm_loss)
        tensorboard_logs = {'val_loss': loss, 'perplexity': perplexity}
        return {'val_loss': loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        """Called at the end of validation to aggregate outputs.

        Args:
            outputs (list): The individual outputs of each validation step.

        Returns:
            dict: Validation loss and tensorboard logs.
        """
        if outputs:
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            perplexity = torch.stack([x['log']['perplexity'] for x in outputs]).mean()
            tensorboard_logs = {'val_loss': avg_loss, 'perplexity': perplexity}
            logging.info(f"evaluation perplexity {perplexity.item()}")
            return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = (
            self._setup_preprocessed_dataloader(train_data_config)
            if self.tokenizer is None
            else self._setup_dataloader(train_data_config)
        )

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = (
            self._setup_preprocessed_dataloader(val_data_config)
            if self.tokenizer is None
            else self._setup_dataloader(val_data_config)
        )

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        pass

    def _setup_preprocessed_dataloader(self, cfg: Optional[DictConfig]):
        dataset = cfg.data_file
        max_predictions_per_seq = cfg.max_predictions_per_seq
        batch_size = cfg.batch_size

        if os.path.isdir(dataset):
            files = [os.path.join(dataset, f) for f in os.listdir(dataset) if os.path.isfile(os.path.join(dataset, f))]
        else:
            files = [dataset]
        files.sort()
        dl = BertPretrainingPreprocessedDataloader(
            data_files=files, max_predictions_per_seq=max_predictions_per_seq, batch_size=batch_size
        )
        return dl

    def _setup_tokenizer(self, cfg: DictConfig):
        tokenizer = get_tokenizer(**cfg)
        self.tokenizer = tokenizer

    def _setup_dataloader(self, cfg: DictConfig):
        dataset = BertPretrainingDataset(
            tokenizer=self.tokenizer,
            data_file=cfg.data_file,
            max_seq_length=cfg.max_seq_length,
            mask_prob=cfg.mask_prob,
            short_seq_prob=cfg.short_seq_prob,
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
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass
