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

import os
from pathlib import Path
from typing import Literal, TypedDict

import torch
from cosmos1.models.autoregressive.nemo.post_training.action_control.prepare_dataset import get_default_output_prefix
from torch.utils.data import Dataset


class ActionControlRow(TypedDict):
    """A single item loaded via the action-control post-training dataset."""

    current_frame: torch.Tensor  # shape: (1, H, W)
    next_frame: torch.Tensor  # shape: (1, H, W)
    action: torch.Tensor  # shape: (7,)


class ActionControlDataset(Dataset):
    """The action-control autoregressive post-training dataset."""

    def __init__(
        self,
        data_path: str | os.PathLike | None = None,
        subfolder: str | None = None,
        split: Literal["train", "val", "test"] = "train",
        seed: int = 42,
        shuffle: bool = True,
    ):
        """Initialize the action-control autoregressive post-training dataset.

        Args:
            data_path: The path to the data. If not provided, this will assume the data is stored in the
                default location in the huggingface cache.
            subfolder: The subfolder to use in HF_HOME/assets/cosmos/action-control. Should not be provided
                if data_path is provided.
            split: The split to use.
        """

        if data_path is None:
            self.data_path = get_default_output_prefix(split, subfolder)
        else:
            assert subfolder is None, "subfolder should not be provided if data_path is provided"
            self.data_path = Path(data_path)

        self.tokenized_frames = torch.load(self.data_path / "tokenized-frames.pt", mmap=True)
        self.actions = torch.load(self.data_path / "actions.pt")

        # Here we use the NaN mask created in prepare_dataset.py to filter out the final frames
        # from each trajectory, as they can't be used as an input action.
        self.valid_actions_indices = torch.where(~torch.isnan(self.actions).all(dim=1))[0]

        # Shuffle the training data.
        if shuffle:
            g = torch.Generator()
            g.manual_seed(seed)
            self.valid_actions_indices = self.valid_actions_indices[
                torch.randperm(len(self.valid_actions_indices), generator=g)
            ]

    def __len__(self) -> int:
        """The number of valid actions in the dataset.

        Since the last frame from each trajectory can't be used as an input action, this is less
        than the total number of frames.
        """
        return len(self.valid_actions_indices)

    def __getitem__(self, i: int) -> ActionControlRow:
        """Get the i-th action-control batch from the dataset.

        Args:
            i: The index of the batch to get.

        Returns:
            A dictionary containing the current tokenized frame, next tokenized frame, and action.
        """
        frame_indices = self.valid_actions_indices[i]
        return ActionControlRow(
            current_frame=self.tokenized_frames[frame_indices],  # shape: (1, H, W)
            next_frame=self.tokenized_frames[frame_indices + 1],  # shape: (1, H, W)
            action=self.actions[frame_indices],  # shape: (7,)
        )
