# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

from nemo.collections.vlm.neva.data.multimodal_tokens import IMAGE_TOKEN_INDEX
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging

from nemo.collections.vlm.openvla.data.prismatic.vla.datasets import DummyDataset
from nemo.collections.vlm.openvla.data.prismatic.vla.action_tokenizer import ActionTokenizer
from nemo.collections.vlm.openvla.data.prismatic.util.data_utils import PaddedCollatorForActionPrediction
from nemo.collections.vlm.openvla.data.prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform


class MockDataModule(pl.LightningDataModule):
    def __init__(
        self,
        seq_length: int = 2048,
        decoder_seq_length: Optional[int] = None,
        tokenizer: Optional = None,
        image_processor: Optional = None,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        num_train_samples: int = 10_000_000,
        num_val_samples: int = 10_000_000,
        num_test_samples: int = 10_000_000,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        # additional params for OpenVLA
        llm_backbone_id: str = "llama2-7b-pure",
        vision_backbone_id: str = "dinosiglip-vit-so-224px",
        llm_max_length: int = 2048, 
        load_for_training: bool = False,
        image_resize_strategy: str = "resize-naive",
        predict_stop_token: bool = True,
        padding_side: str = 'right',
        image_aug: bool = False,    
        train: bool = True,
        hf_token: str = None,
    ):
        super().__init__()

        self.seq_length = seq_length
        self.decoder_seq_len = decoder_seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        # additional params for OpenVLA
        self.llm_backbone_id = llm_backbone_id
        self.vision_backbone_id = vision_backbone_id
        self.llm_max_length = llm_max_length
        self.load_for_training = load_for_training
        self.image_resize_strategy = image_resize_strategy
        self.predict_stop_token = predict_stop_token
        self.padding_side = padding_side
        self.image_aug = image_aug
        self.train = train
        self.hf_token = hf_token

        self.llm_backbone, self.tokenizer = get_llm_backbone_and_tokenizer(
            self.llm_backbone_id,
            llm_max_length=self.llm_max_length,
            hf_token=self.hf_token,
            inference_mode=not self.load_for_training,
        )
        self.vision_backbone, self.image_transform = get_vision_backbone_and_transform(
            self.vision_backbone_id,
            self.image_resize_strategy,
        )

        self.action_tokenizer = ActionTokenizer(self.tokenizer)

        self.collator = PaddedCollatorForActionPrediction(
            self.tokenizer.model_max_length, 
            self.tokenizer.pad_token_id, 
            padding_side=self.padding_side,
        )

        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            decoder_seq_len=self.decoder_seq_len,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        self._train_ds = DummyDataset(
            action_tokenizer = self.action_tokenizer, 
            base_tokenizer = self.tokenizer, 
            image_transform = self.image_transform, 
            prompt_builder_fn = self.llm_backbone.prompt_builder_fn
        )
        self._validation_ds = DummyDataset(
            action_tokenizer = self.action_tokenizer, 
            base_tokenizer = self.tokenizer, 
            image_transform = self.image_transform, 
            prompt_builder_fn = self.llm_backbone.prompt_builder_fn
        )
        self._test_ds = DummyDataset(
            action_tokenizer = self.action_tokenizer, 
            base_tokenizer = self.tokenizer, 
            image_transform = self.image_transform, 
            prompt_builder_fn = self.llm_backbone.prompt_builder_fn
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if not hasattr(self, "_train_ds"):
            self.setup()
        return self._create_dataloader(self._train_ds)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        if not hasattr(self, "_validation_ds"):
            self.setup()
        return self._create_dataloader(self._validation_ds)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if not hasattr(self, "_test_ds"):
            self.setup()
        return self._create_dataloader(self._test_ds)

    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collator,
            **kwargs,
        )
