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
import json
import webdataset as wd
import io
import pickle
from torch.utils.data import Dataset,IterableDataset

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.data_utils import dataset_to_ids
from nemo.utils import logging

__all__ = ['L2RLanguageModelingDataset', 'TarredL2RLanguageModelingDataset']


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
        cache_ids: bool = False
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.batch_step = batch_step or self.max_seq_length
        ids = dataset_to_ids(dataset, tokenizer, cache_ids=cache_ids, add_bos_eos=False)
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


class TarredL2RLanguageModelingDataset(IterableDataset):
    def __init__(
        self,
        tarpath: str,
        metadata_path: str,
        tokenizer,
        shuffle_n: int = 512,
        max_seq_length: int = 512,
        batch_step: int = None,
    ):
        super(TarredL2RLanguageModelingDataset, self).__init__()

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.batch_step = batch_step or self.max_seq_length

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.metadata = metadata

        if isinstance(tarpath, str):
            # Replace '(' and '[' with '{'
            brace_keys_open = ['(', '[', '<', '_OP_']
            for bkey in brace_keys_open:
                if bkey in tarpath:
                    tarpath = tarpath.replace(bkey, "{")

            # Replace ')' and ']' with '}'
            brace_keys_close = [')', ']', '>', '_CL_']
            for bkey in brace_keys_close:
                if bkey in tarpath:
                    tarpath = tarpath.replace(bkey, "}")

        self.tarpath = tarpath

        # Put together WebDataset
        self._dataset = (
            wd.Dataset(tarpath)
            .shuffle(shuffle_n)
            .rename(npy='npy', key='__key__')
            .to_tuple('npy', 'key')
            .map(f=self._build_sample)
        )

    def _build_sample(self, tup):
        npy, filepath = tup
        npy = io.BytesIO(npy)
        data = np.load(npy)
        npy.close()

        # flatten data
        idx = np.random.randint(0, (len(data) - self.max_seq_length) // self.batch_step + 1)

        # random slice of data
        left = idx * self.batch_step
        right = left + self.max_seq_length
        data = data[left:right + 1]
        src_ids = data[:-1]
        labels = data[1:]
        src_mask = (src_ids != self.tokenizer.pad_id).astype(np.float32)
        return src_ids, src_mask, labels

    def __iter__(self):
        return self._dataset.__iter__()

    def __len__(self):
        return (self.metadata['num_text'] - self.max_seq_length) // self.batch_step
