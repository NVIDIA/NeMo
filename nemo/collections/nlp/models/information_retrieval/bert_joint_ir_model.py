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

import math
from typing import Dict, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from nemo.collections.common.losses import SmoothedCrossEntropyLoss
from nemo.collections.nlp.data import BertInformationRetrievalDatasetEval, BertInformationRetrievalDatasetTrain
from nemo.collections.nlp.modules.common import SequenceClassifier
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.classes.common import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import NeuralType

__all__ = ['BertJointIRModel']


class BertJointIRModel(ModelPT):
    """
    Information retrieval model which jointly encodes both query and passage
    and passes them to BERT encoder followed by a fully-connected layer for
    similarity score prediction.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.sim_score_regressor.output_types

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        self.dataset_cfg = cfg.dataset

        self.tokenizer = get_tokenizer(tokenizer_name=cfg.language_model.pretrained_model_name,)

        super().__init__(cfg=cfg, trainer=trainer)

        self.bert_model = get_lm_model(pretrained_model_name=cfg.language_model.pretrained_model_name,)

        # make vocabulary size divisible by 8 for fast fp16 training
        vocab_size = self.tokenizer.vocab_size
        tokens_to_add = 8 * math.ceil(vocab_size / 8) - vocab_size
        # device = self.bert_model.embeddings.word_embeddings.weight.get_device()
        # print (device)
        zeros = torch.zeros((tokens_to_add, cfg.language_model.hidden_size))  # .to(device=device)
        self.bert_model.embeddings.word_embeddings.weight.data = torch.cat(
            (self.bert_model.embeddings.word_embeddings.weight.data, zeros)
        )

        self.sim_score_regressor = SequenceClassifier(
            hidden_size=cfg.language_model.hidden_size,
            num_classes=1,
            num_layers=1,
            dropout=cfg.language_model.dropout,
            log_softmax=False,
        )

        self.loss = SmoothedCrossEntropyLoss(pad_id=self.tokenizer.pad_id)

        # Optimizer setup needs to happen after all model weights are ready
        self.setup_optimization(cfg.optim)

    @typecheck()
    def forward(self, input_ids, token_type_ids, attention_mask):

        hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        scores = self.sim_score_regressor(hidden_states=hidden_states)

        return scores

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        input_ids, input_mask, input_type_ids = batch
        batch_size, num_passages, seq_length = input_ids.size()

        scores = self(
            input_ids=input_ids.view(-1, seq_length),
            token_type_ids=input_type_ids.view(-1, seq_length),
            attention_mask=input_mask.view(-1, seq_length),
        ).view(batch_size, 1, num_passages)
        scores = torch.log_softmax(scores, dim=-1)

        labels = torch.zeros_like(input_ids[:, :1, 0])
        train_loss = self.loss(logits=scores, labels=labels, output_mask=torch.ones_like(labels))

        tensorboard_logs = {'train_loss': train_loss, 'lr': self._optimizer.param_groups[0]['lr']}
        return {'loss': train_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_mask, input_type_ids, query_id, passage_ids = batch
        batch_size, num_passages, seq_length = input_ids.size()

        scores = self(
            input_ids=input_ids.view(-1, seq_length),
            token_type_ids=input_type_ids.view(-1, seq_length),
            attention_mask=input_mask.view(-1, seq_length),
        ).view(batch_size, 1, num_passages)
        scores = torch.log_softmax(scores, dim=-1)

        labels = torch.zeros_like(input_ids[:, :1, 0])
        val_loss = self.loss(logits=scores, labels=labels, output_mask=torch.ones_like(labels))

        tensorboard_logs = {'val_loss': val_loss}
        return {'val_loss': val_loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_ppl': torch.exp(avg_loss)}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, mode="train")

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, mode="eval")

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, mode="eval")

    def _setup_dataloader_from_config(self, cfg: DictConfig, mode="train"):

        dataset_params = {
            "tokenizer": self.tokenizer,
            "passages": cfg.passages,
            "queries": cfg.queries,
            "query_to_passages": cfg.query_to_passages,
            "max_query_length": self.dataset_cfg.get("max_query_length", 31),
            "max_passage_length": self.dataset_cfg.get("max_passage_length", 190),
        }

        if mode == "train":
            dataset = BertInformationRetrievalDatasetTrain(
                num_negatives=cfg.get("num_negatives", 10), **dataset_params,
            )
        elif mode == "eval":
            dataset = BertInformationRetrievalDatasetEval(
                num_candidates=cfg.get("num_candidates", 10), **dataset_params,
            )
        else:
            raise ValueError("mode should be either train or eval")

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            num_workers=self.dataset_cfg.get("num_workers", 2),
            pin_memory=self.dataset_cfg.get("pin_memory", False),
            drop_last=self.dataset_cfg.get("drop_last", False),
        )

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass
