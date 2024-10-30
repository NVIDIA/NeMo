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

from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils import data
from torch.utils.data import DataLoader, Dataset

from nemo.lightning.pytorch.plugins import MegatronDataSampler


class MockDataModule(pl.LightningDataModule):
    def __init__(
        self,
        image_h: int = 1024,
        image_w: int = 1024,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        num_train_samples: int = 10_000,
        num_val_samples: int = 10_000,
        num_test_samples: int = 10_000,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.image_h = image_h
        self.image_w = image_w
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.data_sampler = MegatronDataSampler(
            seq_len=10,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )

    def setup(self, stage: str = "") -> None:
        self._train_ds = _MockT2IDataset(image_H=1024, image_W=1024, length=self.num_train_samples)
        self._validation_ds = _MockT2IDataset(image_H=1024, image_W=1024, length=self.num_val_samples)
        self._test_ds = _MockT2IDataset(image_H=1024, image_W=1024, length=self.num_test_samples)

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
            **kwargs,
        )


class _MockT2IDataset(Dataset):
    def __init__(
            self,
            image_H,
            image_W,
            length=100000,
            image_key='images',
            txt_key='txt',
            hint_key='hint',
            seq_len=80,
            context_dim=768,
            precached_mode=False
    ):
        super().__init__()
        self.length = length
        self.H = image_H
        self.W = image_W
        self.image_key = image_key
        self.txt_key = txt_key
        self.precached_mode = precached_mode
        if precached_mode:
            #TODO implement this
            raise NotImplementedError
            # self.seq_len = seq_len
            # self.context_dim = context_dim

    def __getitem__(self, index):
        item = {}
        if self.precached_mode:
            pass
        else:
            item[self.image_key] = torch.randn(self.H, self.W, 3)
            item[self.txt_key] = "This is a sample caption input"
            item[self.hint_key] = torch.randn(self.H, self.W, 3)
        return item

    def __len__(self):
        return self.length