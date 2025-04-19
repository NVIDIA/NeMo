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

# pylint: disable=C0115,C0116,C0301

import json
import os
import random
from typing import Dict, Literal

import torch
from torch.utils.data import Dataset

from nemo.collections.llm.gpt.data.mock import MockDataModule


class PosEmb3D:
    def __init__(self, *, max_t=96, max_h=960, max_w=960):
        self.max_t = max_t
        self.max_h = max_h
        self.max_w = max_w
        self.generate_pos_id()

    def generate_pos_id(self):
        self.grid = torch.stack(
            torch.meshgrid(
                torch.arange(self.max_t, device='cpu'),
                torch.arange(self.max_h, device='cpu'),
                torch.arange(self.max_w, device='cpu'),
            ),
            dim=-1,
        )

    def get_pos_id_3d(self, *, t, h, w):
        if t > self.max_t or h > self.max_h or w > self.max_w:
            self.max_t = max(self.max_t, t)
            self.max_h = max(self.max_h, h)
            self.max_w = max(self.max_w, w)
            self.generate_pos_id()
        return self.grid[:t, :h, :w]


class ActionControlDiffusionDataset(Dataset):
    def __init__(
        self,
        data_path: str | os.PathLike | None = None,
        subfolder: str | None = "diffusion",
        split: Literal["train", "val", "test"] = "train",
        dtype: torch.dtype = torch.bfloat16,
        context_seq_len: int = 512,
        crossattn_embedding_size: int = 1024,
        original_video_height: int = 480,
        original_video_width: int = 640,
        fps: int = 5,
        num_frames: int = 1,
    ):
        """Initialize the action-control autoregressive post-training dataset.

        Args:
            data_path: The path to the data. If not provided, this will assume the data is stored in the
                default location in the huggingface cache.
            subfolder: The subfolder to use in HF_HOME/assets/cosmos/action-control. Should not be provided
                if data_path is provided.
            split: The split to use.
            dtype: The universal input datatype to load for the model. Modify with model layer dtype.
            context_seq_len: These are predefined from the t5 text embeddings.
            crossattn_embedding_size: These are the size of the hidden dim of the cross attention blocks.
            original_video_height: Height dimension of the original, un-tokenized video.
            original_video_width: Width dimension of the original, un-tokenized video.
            fps: FPS of the video in Hz.
            num_frames: Number of frames to use in each video.
        """
        from cosmos1.models.autoregressive.nemo.post_training.action_control.action_control_dataset import (
            ActionControlDataset,
        )

        if subfolder is not None:
            self.dataset = ActionControlDataset(subfolder=subfolder, split=split)
        else:
            self.dataset = ActionControlDataset(data_path=data_path, split=split)

        # Video metadata associated with the loaded dataset.
        self.context_seq_len = context_seq_len
        self.crossattn_embedding_size = crossattn_embedding_size
        self.original_video_height = original_video_height
        self.original_video_width = original_video_width
        self.dtype = dtype
        self.fps = torch.tensor([fps] * 1, dtype=self.dtype)
        self.num_frames = torch.tensor([num_frames] * 1, dtype=self.dtype)

    def __len__(self) -> int:
        """The number of valid actions in the dataset.

        Since the last frame from each trajectory can't be used as an input action, this is less
        than the total number of frames.
        """
        return len(self.dataset)

    def __getitem__(self, i: int) -> Dict:
        """Get the i-th action-control batch from the dataset.

        Args:
            i: The index of the batch to get.

        Returns:
            A dictionary containing the current tokenized frame, next tokenized frame, and action.
        """
        data = self.dataset[i]
        # Current frame is of shape (<latent_dim>, <timestamp_dim>, height, width)
        current_frame = data['current_frame'].to(self.dtype)
        # Action is of shape (<action_emb_dim>), which is (7) for Bridge.
        action = data['action'].to(self.dtype)
        # Next frame is of shape (<latent_dim>, <timestamp_dim>, height, width)
        next_frame = data['next_frame'].to(self.dtype)

        # video_latent is the input to the DiT V2W model, and it accepts a tensor of shape (B, L, T=2, H, W).
        # The first frame of the tensor is associated with the current video frame, i.e. the video conditioning latent,
        # and the next frame of the tensor is associated with the next video frame that is predicted by the model.
        # The loss is computed across both the noise of the current video frame and the predicted vs. original next frame.
        video_latent = torch.cat([current_frame, next_frame], dim=-3)  # concat on T dimension
        # Video sequence length = T x H x W.
        seq_len = video_latent.shape[-1] * video_latent.shape[-2] * video_latent.shape[-3]  # W, H, T
        loss_mask = torch.ones(seq_len, dtype=self.dtype)
        noise_latent = torch.rand_like(video_latent, dtype=self.dtype)
        timesteps = torch.randn(1, dtype=self.dtype)
        # Note from Imaginaire/DIR team: we send in all zeros to our text embeddings for action control fine-tuning.
        t5_text_embedding = torch.zeros((self.context_seq_len, self.crossattn_embedding_size), dtype=self.dtype)
        t5_text_mask = torch.ones((self.context_seq_len), dtype=self.dtype)
        image_size = torch.tensor(
            [
                [
                    self.original_video_height,
                    self.original_video_width,
                    self.original_video_height,
                    self.original_video_width,
                ]
            ]
            * 1,
            dtype=self.dtype,
        )
        conditioning_latent = current_frame
        padding_mask = torch.zeros((1, 1, self.original_video_height, self.original_video_width), dtype=self.dtype)

        sample = {
            'video': video_latent,  # tokens. We may not flatten it in the same way. AR model flattens it then
            # offsets by 1 token. We may not wanna do that.
            'noise_latent': noise_latent,
            'timesteps': timesteps,
            't5_text_embeddings': t5_text_embedding,
            't5_text_mask': t5_text_mask,
            "image_size": image_size,
            "fps": self.fps,
            "num_frames": self.num_frames,
            "padding_mask": padding_mask,
            "loss_mask": loss_mask,
            "gt_latent": conditioning_latent,
            "num_condition_t": 1,
            "action": action,
        }

        return sample

    def collate_fn(self, batch):
        """
        A default implementation of a collation function.
        Users should override this method to define custom data loaders.
        """
        return torch.utils.data.dataloader.default_collate(batch)


class VideoFolderDataset(Dataset):
    def __init__(self, root_dir='', cache=True):
        self.root_dir = root_dir
        self.sample_prefixes = self._get_sample_prefixes()
        # if cache:
        #     self._cache = {}
        # else:
        #     self._cache = None

    def _get_sample_prefixes(self):
        all_files = os.listdir(self.root_dir)
        prefixes = set()
        for file in all_files:
            prefix = file.split('.')[0]
            prefixes.add(prefix)
        return sorted(list(prefixes))

    def __len__(self):
        return len(self.sample_prefixes)

    def __getitem__(self, idx):
        # if self._cache is not None and idx in self._cache:
        #     return self._cache[idx]
        prefix = self.sample_prefixes[idx]

        # Load JSON info
        with open(os.path.join(self.root_dir, f"{prefix}.info.json"), 'r') as f:
            info = json.load(f)

        # Load text embeddings
        text_embedding = torch.load(os.path.join(self.root_dir, f"{prefix}.t5_text_embeddings.pth"))

        # Load text mask
        text_mask = torch.load(os.path.join(self.root_dir, f"{prefix}.t5_text_mask.pth"))

        # Load video latent
        video_latent = torch.load(os.path.join(self.root_dir, f"{prefix}.video_latent.pth"))

        # Load conditioning latent
        conditioning_latent_path = os.path.join(self.root_dir, f"{prefix}.conditioning_latent.pth")
        if os.path.exists(conditioning_latent_path):
            conditioning_latent = torch.load(conditioning_latent_path, map_location='cpu')
        else:
            conditioning_latent = None

        t = info['num_frames']
        h = info['height']
        w = info['width']

        seq_len = video_latent.shape[-1] * video_latent.shape[-2] * video_latent.shape[-3]
        loss_mask = torch.ones(seq_len, dtype=torch.bfloat16)
        noise_latent = torch.rand_like(video_latent, dtype=torch.bfloat16)
        timesteps = torch.randn(1)
        # pos_emb = self.pos_emb_3d.get_pos_id_3d(t=t, h=h//p, w=w//p)

        sample = {
            'video': video_latent,
            'noise_latent': noise_latent,
            'timesteps': timesteps,
            't5_text_embeddings': text_embedding,
            't5_text_mask': text_mask,
            # 'pos_ids': pos_emb,
            "image_size": torch.tensor([[h, w, h, w]] * 1, dtype=torch.bfloat16),
            "fps": torch.tensor([info['fps']] * 1, dtype=torch.bfloat16),
            "num_frames": torch.tensor([t] * 1, dtype=torch.bfloat16),
            "padding_mask": torch.zeros((1, 1, h, w), dtype=torch.bfloat16),
            "loss_mask": loss_mask,
            "gt_latent": conditioning_latent,
            "num_condition_t": random.randint(1, 4),
        }

        return sample

    def _collate_fn(self, batch):
        """
        A default implementation of a collation function.
        Users should override this method to define custom data loaders.
        """
        return torch.utils.data.dataloader.default_collate(batch)

    def collate_fn(self, batch):
        """Method that user pass as functor to DataLoader.

        The method optionally performs neural type checking and add types to the outputs.

        Please note, subclasses of Dataset should not implement `input_types`.

        # Usage:
        dataloader = torch.utils.data.DataLoader(
                ....,
                collate_fn=dataset.collate_fn,
                ....
        )

        Returns
        -------
            Collated batch, with or without types.
        """
        return self._collate_fn(batch)


class VideoFolderCameraCtrlDataset(Dataset):
    def __init__(self, root_dir='', cache=True):
        self.root_dir = root_dir
        self.sample_prefixes = self._get_sample_prefixes()

    def _get_sample_prefixes(self):
        all_files = os.listdir(self.root_dir)
        prefixes = set()
        for file in all_files:
            prefix = file.split('.')[0]
            prefixes.add(prefix)
        return sorted(list(prefixes))

    def __len__(self):
        return len(self.sample_prefixes)

    def __getitem__(self, idx):
        # if self._cache is not None and idx in self._cache:
        #     return self._cache[idx]
        prefix = self.sample_prefixes[idx]

        # Load JSON info
        with open(os.path.join(self.root_dir, f"{prefix}.info.json"), 'r') as f:
            info = json.load(f)

        # Load text embeddings
        text_embedding = torch.load(os.path.join(self.root_dir, f"{prefix}.t5_text_embeddings.pth"))

        # Load text mask
        text_mask = torch.load(os.path.join(self.root_dir, f"{prefix}.t5_text_mask.pth"))

        # Load video latent
        video_latent = torch.load(os.path.join(self.root_dir, f"{prefix}.video_latent.pth"))

        # Load conditioning latent
        conditioning_latent_path = os.path.join(self.root_dir, f"{prefix}.conditioning_latent.pth")
        conditioning_latent = torch.load(conditioning_latent_path, map_location='cpu')

        # Load plucker embeddings
        plucker_embeddings_path = os.path.join(self.root_dir, f"{prefix}.plucker_embeddings.pth")
        plucker_embeddings = torch.load(plucker_embeddings_path, map_location='cpu')

        # Load image size
        image_size_path = os.path.join(self.root_dir, f"{prefix}.image_size.pth")
        image_size = torch.load(image_size_path, map_location='cpu')

        # Load padding mask
        padding_mask_path = os.path.join(self.root_dir, f"{prefix}.padding_mask.pth")
        padding_mask = torch.load(padding_mask_path, map_location='cpu')

        t = info['num_frames']

        seq_len = video_latent.shape[-1] * video_latent.shape[-2] * video_latent.shape[-3]
        loss_mask = torch.ones(seq_len, dtype=torch.bfloat16)
        noise_latent = torch.rand_like(video_latent, dtype=torch.bfloat16)
        timesteps = torch.randn(1)
        # pos_emb = self.pos_emb_3d.get_pos_id_3d(t=t, h=h//p, w=w//p)

        sample = {
            'video': video_latent,
            'noise_latent': noise_latent,
            'timesteps': timesteps,
            't5_text_embeddings': text_embedding,
            't5_text_mask': text_mask,
            # 'pos_ids': pos_emb,
            "image_size": image_size,
            "fps": torch.tensor([info['fps']] * 1, dtype=torch.bfloat16),
            "num_frames": torch.tensor([t] * 1, dtype=torch.bfloat16),
            "padding_mask": padding_mask,
            "loss_mask": loss_mask,
            "gt_latent": conditioning_latent,
            "num_condition_t": random.randint(1, 4),
            "plucker_embeddings": plucker_embeddings,
        }

        return sample

    def _collate_fn(self, batch):
        """
        A default implementation of a collation function.
        Users should override this method to define custom data loaders.
        """
        return torch.utils.data.dataloader.default_collate(batch)

    def collate_fn(self, batch):
        """Method that user pass as functor to DataLoader.

        The method optionally performs neural type checking and add types to the outputs.

        Please note, subclasses of Dataset should not implement `input_types`.

        # Usage:
        dataloader = torch.utils.data.DataLoader(
                ....,
                collate_fn=dataset.collate_fn,
                ....
        )

        Returns
        -------
            Collated batch, with or without types.
        """
        return self._collate_fn(batch)


class DiTVideoLatentMockDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, seq_len=21760):
        self.length = num_samples if num_samples > 0 else 1 << 32
        self.seq_len = seq_len
        self.pos_emb_3d = PosEmb3D()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        t = 16
        h = 34
        w = 40
        p = 1
        seq_len = t * h * w
        video_latent = torch.randn(1, 16, t, h, w).to(dtype=torch.uint8)
        loss_mask = torch.ones(seq_len, dtype=torch.bfloat16)
        noise_latent = torch.rand_like(video_latent, dtype=torch.bfloat16)
        timesteps = torch.randn(1)
        text_embedding = torch.randn(512, 1024)
        pos_emb = self.pos_emb_3d.get_pos_id_3d(t=t, h=h // p, w=w // p)

        return {
            'video': video_latent,
            'noise_latent': noise_latent,
            'timesteps': timesteps,
            't5_text_embeddings': text_embedding,
            't5_text_mask': torch.ones(512, dtype=torch.bfloat16),
            'pos_ids': pos_emb,
            "image_size": torch.tensor([[34, 40, 34, 40]] * 1, dtype=torch.bfloat16),
            "fps": torch.tensor([30] * 1, dtype=torch.bfloat16),
            "num_frames": torch.tensor([16] * 1, dtype=torch.bfloat16),
            "padding_mask": torch.zeros((1, 1, 34, 40), dtype=torch.bfloat16),
            "loss_mask": loss_mask,
        }

    def _collate_fn(self, batch):
        """
        A default implementation of a collation function.
        Users should override this method to define custom data loaders.
        """
        return torch.utils.data.dataloader.default_collate(batch)

    def collate_fn(self, batch):
        """Method that user pass as functor to DataLoader.

        The method optionally performs neural type checking and add types to the outputs.

        Please note, subclasses of Dataset should not implement `input_types`.

        # Usage:
        dataloader = torch.utils.data.DataLoader(
                ....,
                collate_fn=dataset.collate_fn,
                ....
        )

        Returns
        -------
            Collated batch, with or without types.
        """
        return self._collate_fn(batch)


class DiTActionDataModule(MockDataModule):
    def __init__(
        self,
        path=None,
        subfolder: str = "diffusion",
        dataset=ActionControlDiffusionDataset,
        dtype=torch.bfloat16,
        context_seq_len: int = 512,
        crossattn_embedding_size: int = 1024,
        original_video_height: int = 480,
        original_video_width: int = 640,
        fps: int = 5,
        num_frames: int = 1,
        *args,
        **kwargs,
    ):
        """
        Instantiate the datamodule. Data is automatically downloaded and cached in HF_HOME,
        which can be modified in ENV.
        Pass an explicit path instead of subfolder to point to an explicit dataset directory path.
        """
        super().__init__(*args, **kwargs)
        self.path = path
        self.subfolder = subfolder
        self.dataset = dataset
        self.dtype = dtype
        self.context_seq_len = context_seq_len
        self.crossattn_embedding_size = crossattn_embedding_size
        self.original_video_height = original_video_height
        self.original_video_width = original_video_width
        self.fps = fps
        self.num_frames = num_frames

        if self.path and self.subfolder:
            raise ValueError("We cannot have path and subfolder...")

    def setup(self, stage: str = "") -> None:
        """
        Build ActionControlDiffusionDatasets.
        """
        # Params.
        params = {
            'data_path': self.path,
            'subfolder': self.subfolder,
            'dtype': self.dtype,
            'context_seq_len': self.context_seq_len,
            'crossattn_embedding_size': self.crossattn_embedding_size,
            'original_video_height': self.original_video_height,
            'original_video_width': self.original_video_width,
            'fps': self.fps,
            'num_frames': self.num_frames,
        }
        self._train_ds = self.dataset(split="train", **params)
        self._validation_ds = self.dataset(split='val', **params)
        self._test_ds = self.dataset(split='test', **params)


class DiTDataModule(MockDataModule):
    def __init__(self, *args, path='', dataset=VideoFolderDataset, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path
        self.dataset = dataset

    def setup(self, stage: str = "") -> None:
        self._train_ds = self.dataset(self.path)
        self._validation_ds = self.dataset(self.path)
        self._test_ds = self.dataset(self.path)


class DiTCameraCtrlDataModule(MockDataModule):
    def __init__(self, *args, path='', dataset=VideoFolderCameraCtrlDataset, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path
        self.dataset = dataset

    def setup(self, stage: str = "") -> None:
        self._train_ds = self.dataset(self.path)
        self._validation_ds = self.dataset(self.path)
        self._test_ds = self.dataset(self.path)
