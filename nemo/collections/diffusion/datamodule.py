import json
import os
import random
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset

from nemo.collections.llm.gpt.data.mock import MockDataModule, _MockGPTDataset


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


class VideoFolderDataset(Dataset):
    def __init__(self, root_dir='/lustre/fsw/portfolios/coreai/users/zeeshanp/jensen_cached_7b_data_v2', cache=True):
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
        # if self._cache is not None:
        #     self._cache[idx] = sample
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


class DiTDataModule(MockDataModule):
    def __init__(self, *args, path='', dataset=VideoFolderDataset, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = path
        self.dataset = dataset

    def setup(self, stage: str = "") -> None:
        self._train_ds = self.dataset(self.path)
        self._validation_ds = self.dataset(self.path)
        self._test_ds = self.dataset(self.path)
