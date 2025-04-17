# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0115,C0116,C0301

import json

import torch
from cosmos1.models.autoregressive.modules.embedding import SinCosPosEmbAxisTE
from cosmos1.models.autoregressive.nemo.cosmos import CosmosConfig
from torch.utils.data import Dataset

from nemo.collections.llm.gpt.data.mock import MockDataModule

TOKENIZER_COMPRESSION_FACTOR = [8, 16, 16]
DATA_RESOLUTION_SUPPORTED = [640, 1024]
NUM_CONTEXT_FRAMES = 33
BOV_TOKEN = 64000
PAD_ID = 64002


class CosmosVideo2WorldDataset(Dataset):
    def __init__(self, data_path, model_config, split="train"):
        self.data_path = data_path
        self.model_config = model_config
        self.split = split
        self.abs_pos_emb = get_abs_pos_embed(model_config, training_type="text_to_video")
        metadata_file = f"{self.data_path}/metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        self.metadata = metadata

    def __len__(self):
        return self.metadata[f"{self.split}_samples"]

    def __getitem__(self, idx):
        prompt_embedding = torch.load(f"{self.data_path}/{self.split}_prompt_{idx}.pt", map_location="cpu").to(
            torch.bfloat16
        )
        video_tokens = torch.load(f"{self.data_path}/{self.split}_video_{idx}.pt", map_location="cpu").to(
            torch.bfloat16
        )
        seq_len = video_tokens.numel() - 1
        assert seq_len == 12864, f"expected seq len 12864 but got {seq_len}"
        attention_mask = (
            torch.tril(torch.ones((seq_len, seq_len), device=video_tokens.device)).unsqueeze(0).to(torch.bool)
        )
        loss_mask = torch.cat([torch.ones(12800), torch.zeros(64)])
        sample = {
            "tokens": video_tokens[:-1].to(torch.int64),
            "position_ids": torch.arange(0, seq_len),
            "attention_mask": attention_mask,
            "labels": video_tokens[1:].to(torch.int64),
            "abs_pos_embed": self.abs_pos_emb,
            "loss_mask": loss_mask,
            "context": prompt_embedding,
        }

        return sample

    def _collate_fn(self, batch):
        op = torch.utils.data.dataloader.default_collate(batch)
        op["attention_mask"] = op["attention_mask"][0, :, :, :].unsqueeze(dim=0)
        op["abs_pos_embed"] = op["abs_pos_embed"][0, :, :, :]
        op["context"] = op["context"].permute(1, 0, 2)
        return op

    def collate_fn(self, batch):
        return self._collate_fn(batch)


def get_abs_pos_embed(model_config: CosmosConfig, training_type: str | None = "text_to_video"):
    pos_emb = SinCosPosEmbAxisTE(
        model_config.hidden_size,
        latent_shape=model_config.latent_shape,
        pad_to_multiple_of=model_config.pad_to_multiple_of,
        device="cpu",
    )
    abs_pos_emb = pos_emb.forward(training_type=training_type)
    abs_pos_emb = abs_pos_emb.transpose(0, 1).contiguous()
    return abs_pos_emb


class CosmosVideo2WorldDataModule(MockDataModule):
    def __init__(self, *args, **kwargs):
        data_path = kwargs["data_path"]
        model_config = kwargs["model_config"]
        del kwargs["data_path"]
        del kwargs["model_config"]
        super().__init__(*args, **kwargs)
        self.dataset = CosmosVideo2WorldDataset
        self.data_path = data_path
        self.model_config = model_config

    def setup(self, stage: str = "") -> None:
        self._train_ds = self.dataset(data_path=self.data_path, model_config=self.model_config, split="train")
        self._validation_ds = self.dataset(data_path=self.data_path, model_config=self.model_config, split="test")
        self._test_ds = self.dataset(data_path=self.data_path, model_config=self.model_config, split="val")
