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
    """
    Dataset as used by the LaserTaggerDataLayer for training, validation, and inference
    pipelines.

    Args:
        preprocessed_data (str): path to preprocessed train/validation/test data
        use_t2t_decoder (bool): whether to use Autoregressive Decoder
    """

    def __init__(self, preprocessed_data, use_t2t_decoder):

        self.examples = preprocessed_data
        self.use_t2t_decoder = use_t2t_decoder

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = torch.Tensor(self.examples[idx].features['input_ids']).long()
        input_mask = torch.Tensor(self.examples[idx].features['input_mask'])
        segment_ids = torch.Tensor(self.examples[idx].features['segment_ids']).long()
        labels_mask = torch.Tensor(self.examples[idx].features['labels_mask'])
        labels = torch.Tensor(self.examples[idx].features['labels']).long()
        return input_ids, input_mask, segment_ids, labels, labels_mask, labels, input_mask
