# Copyright 2020 NVIDIA. All Rights Reserved.
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

import itertools
import os
import pickle
import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from nemo import logging
from nemo.core.classes import Dataset
from nemo.utils.decorators import experimental

__all__ = ['BertPretrainingPreprocessedDataset', 'BertPretrainingPreprocessedDataloader']


def load_h5(input_file):
    return h5py.File(input_file, "r")


@experimental
class BertPretrainingPreprocessedDataset(Dataset):
    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = load_h5(input_file)
        keys = [
            'input_ids',
            'input_mask',
            'segment_ids',
            'masked_lm_positions',
            'masked_lm_ids',
            'next_sentence_labels',
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            input[index].astype(np.int64) for input in self.inputs
        ]

        output_mask = np.zeros_like(input_ids)
        output_ids = input_ids.copy()

        index = self.max_pred_length
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices[0]) != 0:
            index = padded_mask_indices[0][0]

        output_mask[masked_lm_positions[:index]] = 1.0
        output_ids[masked_lm_positions[:index]] = masked_lm_ids[:index]

        # input_mask = np.asarray(input_mask, dtype=np.float32)
        # output_mask = np.asarray(output_mask, dtype=np.float32)
        return (input_ids, segment_ids, input_mask, output_ids, output_mask, next_sentence_labels)


class BertPretrainingPreprocessedDataloader(DataLoader):
    def __init__(self, data_files, max_pred_length, batch_size, seed=42):
        super().__init__(None, batch_size=batch_size)
        self.random = random.Random(seed)
        self.data_files = data_files
        self.max_pred_length = max_pred_length

    # def __len__(self):
    #     return sum([len(load_h5(data_file)['input_ids']) for data_file in self.data_files])//(self.batch_size)

    def __iter__(self):
        self.random.shuffle(self.data_files)
        for data_file in self.data_files:
            train_data = BertPretrainingPreprocessedDataset(input_file=data_file, max_pred_length=self.max_pred_length)
            train_sampler = DistributedSampler(train_data)
            # print("---")
            # print(os.getpid(), train_sampler.rank, train_sampler.num_replicas, train_sampler.num_samples)
            # print("---")
            train_dataloader = DataLoader(
                dataset=train_data, sampler=train_sampler, batch_size=self.batch_size, shuffle=False,
            )
            for x in train_dataloader:
                yield x
