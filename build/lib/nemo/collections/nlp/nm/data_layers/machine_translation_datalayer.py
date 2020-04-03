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

import torch
from torch.utils import data as pt_data

import nemo
from nemo.collections.nlp.data import TranslationDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core import ChannelType, LabelsType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['TranslationDataLayer']


class TranslationDataLayer(TextDataLayer):
    """
    Data layer for neural machine translation from source (src) language to
    target (tgt) language.

    Args:
        tokenizer_src (TokenizerSpec): source language tokenizer
        tokenizer_tgt (TokenizerSpec): target language tokenizer
        dataset_src (str): path to source data
        dataset_tgt (str): path to target data
        tokens_in_batch (int): maximum allowed number of tokens in batches,
            batches will be constructed to minimize the use of <pad> tokens
        clean (bool): whether to use parallel data cleaning such as removing
            pairs with big difference in sentences length, removing pairs with
            the same tokens in src and tgt, etc; useful for training data layer
            and should not be used in evaluation data layer
        dataset_type (Dataset):
                the underlying dataset. Default: TranslationDataset
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        src_ids: indices of tokens which correspond to source sentences
        src_mask: bool tensor with 0s in place of source tokens to be masked
        tgt_ids: indices of tokens which correspond to target sentences
        tgt_mask: bool tensor with 0s in place of target tokens to be masked
        labels: indices of tokens which should be predicted from each of the
            corresponding target tokens in tgt_ids; for standard neural
            machine translation equals to tgt_ids shifted by 1 to the right
        sent_ids: indices of the sentences in a batch; important for
            evaluation with external metrics, such as SacreBLEU

        """
        return {
            "src_ids": NeuralType(('B', 'T'), ChannelType()),
            "src_mask": NeuralType(('B', 'T'), ChannelType()),
            "tgt_ids": NeuralType(('B', 'T'), ChannelType()),
            "tgt_mask": NeuralType(('B', 'T'), ChannelType()),
            "labels": NeuralType(('B', 'T'), LabelsType()),
            "sent_ids": NeuralType(tuple('B'), ChannelType()),
        }

    def __init__(
        self,
        tokenizer_src,
        tokenizer_tgt,
        dataset_src,
        dataset_tgt,
        tokens_in_batch=1024,
        shuffle=False,
        clean=False,
        dataset_type=TranslationDataset,
    ):
        dataset_params = {
            'tokenizer_src': tokenizer_src,
            'tokenizer_tgt': tokenizer_tgt,
            'dataset_src': dataset_src,
            'dataset_tgt': dataset_tgt,
            'tokens_in_batch': tokens_in_batch,
            'clean': clean,
        }
        super().__init__(dataset_type, dataset_params, batch_size=1, shuffle=shuffle)

        if self._placement == nemo.core.DeviceType.AllGpu:
            sampler = pt_data.distributed.DistributedSampler(self._dataset)
        else:
            sampler = None

        self._dataloader = pt_data.DataLoader(
            dataset=self._dataset, batch_size=1, collate_fn=self._collate_fn, shuffle=sampler is None, sampler=sampler
        )

    def _collate_fn(self, x):
        src_ids, src_mask, tgt_ids, tgt_mask, labels, sent_ids = x[0]
        src_ids = torch.Tensor(src_ids).long().to(self._device)
        src_mask = torch.Tensor(src_mask).float().to(self._device)
        tgt_ids = torch.Tensor(tgt_ids).long().to(self._device)
        tgt_mask = torch.Tensor(tgt_mask).float().to(self._device)
        labels = torch.Tensor(labels).long().to(self._device)
        sent_ids = torch.Tensor(sent_ids).long().to(self._device)
        return src_ids, src_mask, tgt_ids, tgt_mask, labels, sent_ids

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader
