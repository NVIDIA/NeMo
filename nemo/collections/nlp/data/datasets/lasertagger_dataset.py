# =============================================================================
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
# =============================================================================

"""Pytorch Dataset for training LaserTagger."""

import torch
from torch.utils.data import Dataset

__all__ = ['LaserTaggerDataset']


class LaserTaggerDataset(Dataset):
    def __init__(self, preprocessed_data):

        self.examples, self.num_examples = torch.load(preprocessed_data)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        input_ids = torch.Tensor(self.examples[idx].features['input_ids']).long()
        input_mask = torch.Tensor(self.examples[idx].features['input_mask'])
        segment_ids = torch.Tensor(self.examples[idx].features['segment_ids']).long()
        tgt_ids = torch.Tensor(self.examples[idx].features['tgt_ids']).long()
        labels_mask = torch.Tensor(self.examples[idx].features['labels_mask'])
        labels = torch.Tensor(self.examples[idx].features['labels']).long()
        task = self.examples[idx].editing_task
        return input_ids, input_mask, segment_ids, tgt_ids, labels_mask, labels  # , task
