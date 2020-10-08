# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.data_utils import dataset_to_ids

__all__ = ['L2RLanguageModelingDataset']


class L2RLanguageModelingDataset(Dataset):
    """
    Dataset for training and evaluating left-to-right language models.
    
    Args:
        tokenizer: tokenizer, such as WordTokenizer or CharTokenizer
        dataset: path to data
        max_seq_length: maximum sequence length (in tokens) of input tensors
        batch_step: distance (in tokens) between two successive sequences of
            the text. By default, it is equal to max_seq_length which corresponds
            to splitting text into disjoint segments covering full dataset
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        dataset: str,
        max_seq_length: Optional[int] = 512,
        batch_step: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.batch_step = batch_step or self.max_seq_length
        ids = dataset_to_ids(dataset, tokenizer, add_bos_eos=False)
        self.ids = np.array([j for i in ids for j in i])

    def __len__(self):
        return (len(self.ids) - self.max_seq_length) // self.batch_step

    def __getitem__(self, idx):
        left = idx * self.batch_step
        right = left + self.max_seq_length
        src_ids = self.ids[left:right]
        labels = self.ids[left + 1 : right + 1]
        src_mask = (src_ids != self.tokenizer.pad_id).astype(np.float32)
        return src_ids, src_mask, labels
