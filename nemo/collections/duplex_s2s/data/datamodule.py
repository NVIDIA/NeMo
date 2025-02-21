# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import torch
from lightning import LightningDataModule

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.duplex_s2s.data.dataset import DuplexS2SDataset


class S2SDataModule(LightningDataModule):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = ...  # TODO(pzelasko): tokenizer

    def train_dataloader(self):
        if "train_ds" not in self.cfg:
            return None
        return get_lhotse_dataloader_from_config(
            config=self.cfg.train_ds,
            global_rank=torch.distributed.get_rank(),
            world_size=torch.distributed.get_world_size(),
            dataset=DuplexS2SDataset(),
            tokenizer=self.tokenizer,
        )

    def val_dataloader(self):
        # TODO(pzelasko): multi-dataloader
        if "validation_ds" not in self.cfg:
            return None
        return get_lhotse_dataloader_from_config(
            config=self.cfg.validation_ds,
            global_rank=torch.distributed.get_rank(),
            world_size=torch.distributed.get_world_size(),
            dataset=DuplexS2SDataset(),
            tokenizer=self.tokenizer,
        )

    def test_dataloader(self):
        # TODO(pzelasko): multi-dataloader
        if "test_ds" not in self.cfg:
            return None
        return get_lhotse_dataloader_from_config(
            config=self.cfg.test_ds,
            global_rank=torch.distributed.get_rank(),
            world_size=torch.distributed.get_world_size(),
            dataset=DuplexS2SDataset(),
            tokenizer=self.tokenizer,
        )
