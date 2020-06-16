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
    def __init__(self, preprocessed_data, tokenizer, num_examples, training):

        self.examples, self.num_examples = preprocessed_data, num_examples
        self.tokenizer = tokenizer
        self.training = training

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        input_ids = torch.Tensor(self.examples[idx].features['input_ids']).long()
        input_mask = torch.Tensor(self.examples[idx].features['input_mask'])
        segment_ids = torch.Tensor(self.examples[idx].features['segment_ids']).long()
        tgt_ids = torch.Tensor(self.examples[idx].features['tgt_ids']).long()
        labels_mask = torch.Tensor(self.examples[idx].features['labels_mask'])
        labels = torch.Tensor(self.examples[idx].features['labels']).long()

        src_ids, src_first_tokens = torch.zeros([1]), torch.zeros([1])
        if not self.training:
            source_tokens = self.examples[idx].editing_task.source_tokens
            src_ids = self.tokenizer.tokens_to_ids(source_tokens)
            src_ids.extend([self.tokenizer.pad_id] * (128 - len(src_ids)))
            first_tokens = self.examples[idx].editing_task.first_tokens
            first_tokens.extend([-1] * (2 - len(first_tokens)))
            src_ids = torch.Tensor(src_ids).long()
            src_first_tokens = torch.Tensor(first_tokens).long()
        return input_ids, input_mask, segment_ids, tgt_ids, labels_mask, labels, labels_mask, src_ids, src_first_tokens
