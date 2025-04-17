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

import os
from typing import Literal, TypedDict

import torch
from cosmos1.models.autoregressive.nemo.cosmos import CosmosConfig
from cosmos1.models.autoregressive.nemo.post_training.action_control.action_control_dataset import ActionControlDataset
from cosmos1.models.autoregressive.nemo.post_training.video2world_dataset import get_abs_pos_embed

from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.core.classes.dataset import Dataset

BOV_TOKEN = 64000
PAD_ID = 64002


class ActionControlARBatch(TypedDict):
    """A processed batch of data to include additonal keys needed by the Video2World model."""

    tokens: torch.Tensor  # shape: (B, T)
    position_ids: torch.Tensor  # shape: (B, T)
    attention_mask: torch.Tensor  # shape: (B, T, T)
    labels: torch.Tensor  # shape: (B, T)
    abs_pos_embed: torch.Tensor  # shape: (B, T, H, W)
    action: torch.Tensor  # shape: (B, 7)  # Converted to context via MLP
    loss_mask: torch.Tensor  # shape: (B, T)


def ar_collate_fn(batch: list[ActionControlARBatch]) -> ActionControlARBatch:
    op = torch.utils.data.dataloader.default_collate(batch)
    op["attention_mask"] = op["attention_mask"][0, :, :, :].unsqueeze(dim=0)
    op["abs_pos_embed"] = op["abs_pos_embed"][0, :, :, :]
    return op


class ActionControlARDataset(Dataset):
    """The action-control autoregressive post-training dataset."""

    def __init__(
        self,
        model_config: CosmosConfig,
        data_path: str | os.PathLike | None = None,
        subfolder: str | None = None,
        split: Literal["train", "val", "test"] = "train",
    ):
        self._base_ds = ActionControlDataset(data_path=data_path, subfolder=subfolder, split=split)
        self._collate_fn = ar_collate_fn
        self._abs_pos_embed = get_abs_pos_embed(model_config, training_type="text_to_video")

    def __len__(self) -> int:
        return len(self._base_ds)

    def __getitem__(self, i: int) -> ActionControlARBatch:
        row = self._base_ds[i]

        # Stack the starting frame and next frame into a single tensor, prepended with the BOV token
        # and pad it to be divisible by 128.
        tokens = torch.cat(
            (
                torch.Tensor([BOV_TOKEN]),
                row["current_frame"].view(-1),  # shape: (1200,) for DV8x16x16
                row["next_frame"].view(-1),  # shape: (1200,) for DV8x16x16
                torch.Tensor([PAD_ID] * 32),
            ),
            dim=0,
        ).to(torch.int64)

        seq_len = tokens.numel() - 1
        frame_len = row["current_frame"].numel()
        assert seq_len % 128 == 0, "sequence length should be divisible by 128"
        assert seq_len == 2_432, f"sequence length should be 2_432, but is actually {seq_len}"

        attention_mask = (
            torch.tril(torch.ones((seq_len, seq_len), device=row["current_frame"].device)).unsqueeze(0).to(torch.bool)
        )

        loss_mask = torch.zeros(seq_len)
        loss_mask[frame_len : frame_len * 2] = 1  # Only predict the next frame in an autoregressive manner.

        # Like the video2world model, we offset the tokens in the labels by 1, so that the model
        # predicts the next token from the current tokens in an autoregressive manner.
        return {
            "tokens": tokens[:-1],
            "position_ids": torch.arange(0, seq_len, device=row["current_frame"].device),
            "attention_mask": attention_mask,
            "labels": tokens[1:],
            "abs_pos_embed": self._abs_pos_embed,
            "action": row["action"].to(torch.bfloat16),
            "loss_mask": loss_mask,
        }


class ActionControlDataModule(MockDataModule):
    """The action-control autoregressive post-training dataloader."""

    def __init__(
        self,
        *args,
        model_config: CosmosConfig,
        data_path: str | os.PathLike | None = None,
        subfolder: str | None = None,
        **kwargs,
    ):
        """Initialize the action-control autoregressive post-training dataloader.

        Args:
            model_config: The model configuration to use for calculating the absolute position embedding.
            data_path: The path to the data. If not provided, this will assume the data is stored in the
                default location in the huggingface cache.
            subfolder: The subfolder to use in HF_HOME/assets/cosmos/action-control. Should not be provided
                if data_path is provided.
        """
        super().__init__(*args, **kwargs)
        self.data_path = data_path
        self.subfolder = subfolder
        self.model_config = model_config

    def setup(self, stage: str | None = None):
        """Setup the action-control autoregressive post-training dataloader."""
        self._train_ds = ActionControlARDataset(
            data_path=self.data_path,
            subfolder=self.subfolder,
            split="train",
            model_config=self.model_config,
        )
        self._validation_ds = ActionControlARDataset(
            data_path=self.data_path,
            subfolder=self.subfolder,
            split="test",
            model_config=self.model_config,
        )
        self._test_ds = ActionControlARDataset(
            data_path=self.data_path,
            subfolder=self.subfolder,
            split="val",
            model_config=self.model_config,
        )


if __name__ == "__main__":
    import typer
    from cosmos1.models.autoregressive.nemo.cosmos_action_control import CosmosConfigActionControl5B

    def print_shapes(subfolder: str = "autoregressive"):
        example_dataset = ActionControlDataset(split="val", subfolder=subfolder)

        print(f"{len(example_dataset) = }")
        for key, val in example_dataset[0].items():
            print(f"{key = }")
            print(f"{val.shape = }")

        # For the DV8x16x16 tokenizer, the tokenized frames are 30x40.
        # For the CV8x8x8 tokenizer, the tokenized frames are 60x80.
        data_module = ActionControlDataModule(model_config=CosmosConfigActionControl5B(), subfolder=subfolder)
        data_module.setup()
        train_dataloader = data_module.train_dataloader()

        # The dataloader here isn't being wrapped with the MegatronDataSampler, so the batch size hasn't
        # been set to anything other than 1. We can't do that outside a megatron context, so this just
        # demonstrates that iterating over the dataloader works.
        print(f"{len(train_dataloader) = }")
        for batch in train_dataloader:
            for key, val in batch.items():
                print(f"{key = }")
                print(f"{val.shape = }")
            break

    typer.run(print_shapes)
