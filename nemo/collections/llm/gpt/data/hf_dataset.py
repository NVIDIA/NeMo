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

from datasets import load_dataset
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging
from torch.utils.data import DataLoader

import datasets.dataset_dict.DatasetDict
import lightning.pytorch as pl
import torch

def listify(x):
    if isinstance(x, list):
        return x
    return [x]

def extract_split(dataset, split_names):
    if isinstance(dataset, datasets.dataset_dict.DatasetDict):
        for split_name in split_names:
            if split_name in dataset:
                return dataset[split_name]
        raise ValueError(("Dataset does not contain any of " + str(split_names) + \
            "; available splits= " + str(dataset.keys()))
        )
    else:
        return dataset

class HFDatasetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path,
        split=None,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        seq_length=1024,
        micro_batch_size=2,
        global_batch_size=2,
        pad_token_id=0,
        use_mcore_sampler=False,
        mcore_dataloader_type='cyclic',
        **kwargs,
    ) -> None:
        super().__init__()
        assert pad_token_id is not None

        logging.info(f"Loading HF dataset from {path}")

        self.dataset = load_dataset(path, **kwargs)
        if isinstance(self.dataset, datasets.dataset_dict.DatasetDict):
            split_names = self.dataset.keys()
            logging.info(f"HF dataset has the following splits: {split_names}")
        else:
            logging.info(f"Loaded HF dataset has a single split.")


        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.pad_token_id = pad_token_id

        self.use_mcore_sampler = use_mcore_sampler
        self.mcore_dataloader_type = mcore_dataloader_type

    @staticmethod
    def collate_fn(batch, pad_token_id=0):
        def batchify(tensor):
            if tensor.ndim == 1:
                return tensor.unsqueeze_(0)
            return tensor

        def extract_key_from_dicts(batch, key):
            return list(map(lambda x: x[key], batch))

        def pad_within_micro(batch, pad_token_id):
            max_len = max(map(len, batch))
            return [item + [pad_token_id] * (max_len - len(item)) for item in batch]

        keys = list(filter(lambda x: x in batch[0], ['tokens', 'labels', 'position_ids', 'loss_mask']))
        return {
            key: batchify(
                torch.LongTensor(
                    pad_within_micro(
                        extract_key_from_dicts(batch, key),
                        pad_token_id,
                    )
                )
            )
            for key in keys
        }

    def setup(self, stage: str):
        if not self.use_mcore_sampler:
            return
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            dataloader_type=self.mcore_dataloader_type,
        )

    def _make_dataloader(self, dataset, collate_fn=None):
        if collate_fn is None:
            collate_fn = lambda x: HFDatasetDataModule.collate_fn(x, pad_token_id=self.pad_token_id)

        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
            batch_size=self.micro_batch_size,
        )

    def train_dataloader(self, collate_fn=None, split_names=["train", "training"]):
        dataset = extract_split(self.dataset, split_names)
        return self._make_dataloader(dataset, collate_fn)

    def val_dataloader(self, collate_fn=None, split_names=["val", "validation", "eval"]):
        dataset = extract_split(self.dataset, split_names)
        return self._make_dataloader(dataset, collate_fn)

    def test_dataloader(self, collate_fn=None, split_names=["test", "testing"]):
        dataset = extract_split(self.dataset, split_names)
        return self._make_dataloader(dataset, collate_fn)

    def map(self, function=None, split_names=None, **kwargs):
        if split_names is not None:
            datasets = extract_split(self.dataset, split_names)
        else:
            datasets = self.dataset

        if isinstance(dataset, datasets.dataset_dict.DatasetDict):
            dataset_iter = datasets.values()
        else:
            dataset_iter = [datasets]

        for subset in dataset_iter:
            subset.map(function, **kwargs)