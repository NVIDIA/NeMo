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

import os
import random

import h5py
import numpy as np
import torch
from torch.utils import data as pt_data

from nemo.backends.pytorch import DataLayerNM
from nemo.collections.nlp.data import BertPretrainingDataset, BertPretrainingPreprocessedDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import ChannelType, LabelsType, MaskType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['BertPretrainingDataLayer', 'BertPretrainingPreprocessedDataLayer']


class BertPretrainingDataLayer(TextDataLayer):
    """
    Data layer for masked language modeling task for text data.

    Args:
        tokenizer (TokenizerSpec): tokenizer
        dataset (str): directory or a single file with dataset documents
        max_seq_length (int): maximum allowed length of the text segments
        mask_probability (float): probability of masking input sequence tokens
        batch_size (int): batch size in segments
        short_seeq_prob (float): Probability of creating sequences which are
            shorter than the maximum length.
            Defaults to 0.1.
        shuffle (bool): whether to shuffle data or not. Default: False.
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        input_ids:
            indices of tokens which constitute batches of masked text segments
        input_type_ids:
            tensor with 0's and 1's to denote the text segment type
        input_mask:
            bool tensor with 0s in place of tokens to be masked
        output_ids: indices of tokens which constitute batches of unmasked text segments
        output_mask: bool tensor with 0s in place of tokens to be masked
        labels: 0 or 1 for next sentence prediction classification
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_type_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), ChannelType()),
            "output_ids": NeuralType(('B', 'T'), LabelsType()),
            "output_mask": NeuralType(('B', 'T'), MaskType()),
            "labels": NeuralType(tuple('B'), LabelsType()),
        }

    def __init__(
        self, tokenizer, dataset, max_seq_length, mask_probability, short_seq_prob=0.1, batch_size=64, shuffle=False
    ):
        dataset_params = {
            'tokenizer': tokenizer,
            'dataset': dataset,
            'max_seq_length': max_seq_length,
            'mask_probability': mask_probability,
            'short_seq_prob': short_seq_prob,
        }
        super().__init__(BertPretrainingDataset, dataset_params, batch_size, shuffle=shuffle)


class BertPretrainingPreprocessedDataLayer(DataLayerNM):
    """
    Data layer for masked language modeling task for preprocessed data.

    Args:
        dataset (str): directory or a single file with dataset documents
        max_seq_length (int): maximum allowed length of the text segments
        batch_size (int): batch size in segments
        mode (str): model execution mode, e.g. "training"
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        input_ids:
            indices of tokens which constitute batches of masked text segments
        input_type_ids:
            tensor with 0's and 1's to denote the text segment type
        input_mask:
            bool tensor with 0s in place of tokens to be masked
        output_ids: indices of tokens which constitute batches of unmasked text segments
        output_mask: bool tensor with 0s in place of tokens to be masked
        labels: 0 or 1 for next sentence prediction classification
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_type_ids": NeuralType(('B', 'T'), ChannelType()),
            "input_mask": NeuralType(('B', 'T'), ChannelType()),
            "output_ids": NeuralType(('B', 'T'), LabelsType()),
            "output_mask": NeuralType(('B', 'T'), MaskType()),
            "labels": NeuralType(tuple('B'), LabelsType()),
        }

    def __init__(
        self, dataset, max_pred_length, mode, batch_size=64,
    ):
        super().__init__()
        if os.path.isdir(dataset):
            self.files = [
                os.path.join(dataset, f) for f in os.listdir(dataset) if os.path.isfile(os.path.join(dataset, f))
            ]
        else:
            self.files = [dataset]
        self.files.sort()
        self.num_files = len(self.files)
        self._batch_size = batch_size
        self.max_pred_length = max_pred_length
        self.mode = mode
        total_length = 0
        for f in self.files:
            fp = h5py.File(f, 'r')
            total_length += len(fp['input_ids'])
            fp.close()
        self.total_length = total_length

    def _collate_fn(self, x):
        num_components = len(x[0])
        components = [[] for _ in range(num_components)]
        batch_size = len(x)
        for i in range(batch_size):
            for j in range(num_components):
                components[j].append(x[i][j])
        src_ids, src_segment_ids, src_mask, tgt_ids, tgt_mask, sent_ids = [np.stack(x, axis=0) for x in components]
        src_ids = torch.Tensor(src_ids).long().to(self._device)
        src_segment_ids = torch.Tensor(src_segment_ids).long().to(self._device)
        src_mask = torch.Tensor(src_mask).long().to(self._device)
        tgt_ids = torch.Tensor(tgt_ids).long().to(self._device)
        tgt_mask = torch.Tensor(tgt_mask).long().to(self._device)
        sent_ids = torch.Tensor(sent_ids).long().to(self._device)
        return src_ids, src_segment_ids, src_mask, tgt_ids, tgt_mask, sent_ids

    def __len__(self):
        return self.total_length

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        while True:
            if self.mode == "train":
                random.shuffle(self.files)
            for f_id in range(self.num_files):
                data_file = self.files[f_id]
                train_data = BertPretrainingPreprocessedDataset(
                    input_file=data_file, max_pred_length=self.max_pred_length
                )
                train_sampler = pt_data.RandomSampler(train_data)
                train_dataloader = pt_data.DataLoader(
                    dataset=train_data,
                    batch_size=self._batch_size,
                    collate_fn=self._collate_fn,
                    shuffle=False,
                    sampler=train_sampler,
                )
                for x in train_dataloader:
                    yield x
            if self.mode != "train":
                break
