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
from typing import Dict, List, Optional, Union

import onnx
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.nlp.data import SGDDataset
from nemo.collections.nlp.data.dialogue_state_tracking_sgd import Schema, SGDDataProcessor
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.modules.common import TokenClassifier
from nemo.collections.nlp.modules.common.lm_utils import get_lm_model
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import NeuralType
from nemo.utils import logging
from nemo.utils.export_utils import attach_onnx_to_onnx

__all__ = ['SGDQAModel']


class SGDQAModel(NLPModel, Exportable):
    """Dialogue State Tracking Model SGD-QA"""

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.classifier.output_types

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):

        self.setup_tokenizer(cfg.tokenizer)
        super().__init__(cfg=cfg, trainer=trainer)
        self.bert_model = get_lm_model(
            pretrained_model_name=cfg.language_model.pretrained_model_name,
            config_file=cfg.language_model.config_file,
            config_dict=OmegaConf.to_container(cfg.language_model.config) if cfg.language_model.config else None,
            checkpoint_file=cfg.language_model.lm_checkpoint,
        )

        # self.encoder = SGDEncoder(hidden_size=self.bert_model.config.hidden_size, dropout=self._cfg.encoder.dropout)
        # self.decoder = SGDDecoder(embedding_dim=self.bert_model.config.hidden_size)
        # self.loss = SGDDialogueStateLoss(reduction="mean")

    @typecheck()
    def forward(self, input_ids, token_type_ids, attention_mask):
        pass

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        import ipdb

        ipdb.set_trace()
        # input_ids, input_type_ids, input_mask, subtokens_mask, loss_mask, labels = batch
        # logits = self(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        loss = 0
        lr = self._optimizer.param_groups[0]['lr']

        self.log('train_loss', loss)
        self.log('lr', lr, prog_bar=True)

        return {
            'loss': loss,
            'lr': lr,
        }

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        # input_ids, input_type_ids, input_mask, subtokens_mask, loss_mask, labels = batch
        # logits = self(input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask)
        val_loss = 0

        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        outputs: list of individual outputs of each validation step.
        """
        avg_loss = 0

        self.log('val_loss', avg_loss, prog_bar=True)

    def prepare_data(self):
        schema_config = {
            "MAX_NUM_CAT_SLOT": 6,
            "MAX_NUM_NONCAT_SLOT": 12,
            "MAX_NUM_VALUE_PER_CAT_SLOT": 12,
            "MAX_NUM_INTENT": 4,
            "NUM_TASKS": 6,
            "MAX_SEQ_LENGTH": self._cfg.dataset.max_seq_length,
        }
        all_schema_json_paths = []
        for dataset_split in ['train', 'test', 'dev']:
            all_schema_json_paths.append(os.path.join(self._cfg.dataset.data_dir, dataset_split, "schema.json"))
        schemas = Schema(all_schema_json_paths)
        self.dialogues_processor = SGDDataProcessor(
            task_name=self._cfg.dataset.task_name,
            data_dir=self._cfg.dataset.data_dir,
            dialogues_example_dir=self._cfg.dataset.dialogues_example_dir,
            tokenizer=self.tokenizer,
            schemas=schemas,
            schema_config=schema_config,
            subsample=self._cfg.dataset.subsample,
            overwrite_dial_files=not self._cfg.dataset.use_cache,
        )

    def setup_training_data(self, train_data_config: Optional[DictConfig] = None):
        self.prepare_data()
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, split='train')

    def setup_validation_data(self, val_data_config: Optional[DictConfig] = None):
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, split='dev')

    def _setup_dataloader_from_config(self, cfg: DictConfig, split: str) -> DataLoader:
        """
        Setup dataloader from config
        Args:
            cfg: config for the dataloader
        Return:
            Pytorch Dataloader
        """
        dataset_cfg = self._cfg.dataset
        data_dir = dataset_cfg.data_dir

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory is not found at: {data_dir}.")

        dataset = SGDDataset(dataset_split=split, dialogues_processor=self.dialogues_processor)

        dl = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            collate_fn=dataset.collate_fn,
            drop_last=cfg.drop_last,
            shuffle=cfg.shuffle,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
        return dl

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        pass
