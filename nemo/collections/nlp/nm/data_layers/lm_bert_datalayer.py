# =============================================================================
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
from nemo.core import AxisType, BatchTag, NeuralType, TimeTag

__all__ = ['BertPretrainingDataLayer', 'BertPretrainingPreprocessedDataLayer']


class BertPretrainingDataLayer(TextDataLayer):
    """
    Data layer for masked language modeling task.

    Args:
        tokenizer (TokenizerSpec): tokenizer
        dataset (str): directory or a single file with dataset documents
        max_seq_length (int): maximum allowed length of the text segments
        mask_probability (float): probability of masking input sequence tokens
        batch_size (int): batch size in segments
        short_seeq_prob (float): Probability of creating sequences which are
            shorter than the maximum length.
            Defualts to 0.1.
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids: indices of tokens which constitute batches of text segments
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_type_ids: indices of token types (e.g., sentences A & B in BERT)
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_mask: bool tensor with 0s in place of tokens to be masked
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        output_ids: indices of output tokens which should be predicted
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        output_mask: bool tensor with 0s in place of tokens to be excluded
            from loss calculation
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        labels: indices of classes to be predicted from [CLS] token of text
            segments (e.g, 0 or 1 in next sentence prediction task)
            0: AxisType(BatchTag)

        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "output_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "output_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "labels": NeuralType({0: AxisType(BatchTag)}),
        }

    def __init__(self, tokenizer, dataset, max_seq_length, mask_probability, short_seq_prob=0.1, batch_size=64):
        dataset_params = {
            'tokenizer': tokenizer,
            'dataset': dataset,
            'max_seq_length': max_seq_length,
            'mask_probability': mask_probability,
            'short_seq_prob': short_seq_prob,
        }
        super().__init__(BertPretrainingDataset, dataset_params, batch_size, shuffle=False)


class BertPretrainingPreprocessedDataLayer(DataLayerNM):
    """
    Data layer for masked language modeling task.

    Args:
        tokenizer (TokenizerSpec): tokenizer
        dataset (str): directory or a single file with dataset documents
        max_seq_length (int): maximum allowed length of the text segments
        mask_probability (float): probability of masking input sequence tokens
        batch_size (int): batch size in segments
        short_seeq_prob (float): Probability of creating sequences which are
            shorter than the maximum length.
            Defualts to 0.1.
    """

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        input_ids: indices of tokens which constitute batches of text segments
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_type_ids: indices of token types (e.g., sentences A & B in BERT)
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        input_mask: bool tensor with 0s in place of tokens to be masked
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        output_ids: indices of output tokens which should be predicted
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        output_mask: bool tensor with 0s in place of tokens to be excluded
            from loss calculation
            0: AxisType(BatchTag)

            1: AxisType(TimeTag)

        labels: indices of classes to be predicted from [CLS] token of text
            segments (e.g, 0 or 1 in next sentence prediction task)
            0: AxisType(BatchTag)

        """
        return {
            "input_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_type_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "input_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "output_ids": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "output_mask": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            "labels": NeuralType({0: AxisType(BatchTag)}),
        }

    def __init__(self, dataset, max_pred_length, batch_size=64, training=True):

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
        self.training = training
        total_length = 0
        for f in self.files:
            fp = h5py.File(f, 'r')
            total_length += len(fp['input_ids'])
            fp.close()
        self.total_length = total_length
        super().__init__()

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
            if self.training:
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
