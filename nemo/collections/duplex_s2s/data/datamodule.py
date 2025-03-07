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
from omegaconf import open_dict

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.duplex_s2s.data.dataset import DuplexS2SDataset


class S2SDataModule(LightningDataModule):
    def __init__(self, cfg, tokenizer: TokenizerSpec) -> None:
        super().__init__()
        self.cfg = cfg
        with open_dict(self.cfg):
            for k in ("validation_ds", "test_ds"):
                if k in self.cfg:
                    getattr(self.cfg, k).force_finite = True
                    getattr(self.cfg, k).force_map_dataset = True
        self.tokenizer = tokenizer
        self.dataset = DuplexS2SDataset(self.tokenizer, self.cfg.frame_length, self.cfg.source_sample_rate)

    def train_dataloader(self):
        if "train_ds" not in self.cfg:
            return None
        return get_lhotse_dataloader_from_config(
            config=self.cfg.train_ds,
            global_rank=self._get_dp_rank(),
            world_size=self._get_world_size(),
            dataset=self.dataset,
            tokenizer=self.tokenizer,
        )

    def val_dataloader(self):
        # TODO(pzelasko): multi-dataloader
        if "validation_ds" not in self.cfg:
            return None
        return get_lhotse_dataloader_from_config(
            config=self.cfg.validation_ds,
            global_rank=self._get_dp_rank(),
            world_size=self._get_world_size(),
            dataset=self.dataset,
            tokenizer=self.tokenizer,
        )

    def test_dataloader(self):
        # TODO(pzelasko): multi-dataloader
        if "test_ds" not in self.cfg:
            return None
        self.cfg.test_ds.force_finite = True
        self.cfg.test_ds.force_map_dataset = True
        return get_lhotse_dataloader_from_config(
            config=self.cfg.test_ds,
            global_rank=self._get_dp_rank(),
            world_size=self._get_world_size(),
            dataset=self.dataset,
            tokenizer=self.tokenizer,
        )

    def _get_dp_rank(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if (
                hasattr(self.trainer.model, "device_mesh") and self.trainer.model.device_mesh is not None
            ):  # model parallelism
                return self.trainer.model.device_mesh.get_coordinate()[0]
            else:
                return torch.distributed.get_rank()  # plain ol' DDP
        else:
            return 0  # 1 GPU

    def _get_world_size(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if (
                hasattr(self.trainer.model, "device_mesh") and self.trainer.model.device_mesh is not None
            ):  # model parallelism
                return self.trainer.model.device_mesh.shape[0]
            else:  # plain ol' DDP
                return torch.distributed.get_world_size()
        else:
            return 1  # 1 GPU
