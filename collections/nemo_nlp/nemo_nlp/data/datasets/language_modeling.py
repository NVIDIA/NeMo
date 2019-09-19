# Copyright 2019 AI Applications Design Team at NVIDIA. All Rights Reserved.
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
# ==============================================================================
"""Pytorch Dataset for training Neural Machine Translation."""

import numpy as np
from torch.utils.data import Dataset

from ..utils import dataset_to_ids


class LanguageModelingDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 dataset,
                 max_sequence_length=512,
                 batch_step=None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_sequence_length
        self.batch_step = batch_step or self.max_seq_length
        ids = dataset_to_ids(dataset, tokenizer, add_bos_eos=False)
        self.ids = np.array([j for i in ids for j in i])

    def __len__(self):
        return (len(self.ids) - self.max_seq_length) // self.batch_step

    def __getitem__(self, idx):
        left = idx * self.batch_step
        right = left + self.max_seq_length
        src_ids = self.ids[left:right]
        labels = self.ids[left + 1:right + 1]
        src_mask = (src_ids != self.tokenizer.pad_id()).astype(np.float32)
        return src_ids, src_mask, labels
