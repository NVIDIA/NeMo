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

# =============================================================================
# Copyright 2019 Salesforce Research.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom
# the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
# THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# =============================================================================

import numpy as np
import torch
from torch.utils import data as pt_data

import nemo
from nemo.collections.nlp.data.datasets import MultiWOZDataset
from nemo.collections.nlp.nm.data_layers.text_datalayer import TextDataLayer
from nemo.core.neural_types import ChannelType, LabelsType, LengthsType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['MultiWOZDataLayer']


class MultiWOZDataLayer(TextDataLayer):
    """
    Creates the data layer to use for State Tracking dataset MultiWOZ.

    Args:
        data_dir (str): path of the data folder
        domains (dict): dictionary of the domains to include
        all_domains (dict): dictionary of all the available domains
        vocab (Vocab): Vocabulary
        slots (list): list of the slots
        gating_dict (dict): dictionary of the gates
        num_samples (int): number of samples to include
        batch_size (int): batch size
        mode (str): mode of dataset, default='train'
        dataset_type (Dataset): Dataset Type
        shuffle (bool): enables shuffling, default=False
        num_workers (int): number of workers
        input_dropout (float): dropout applied to the input
        is_training (bool): specifies if it is for training
    """

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        src_ids: ids of input sequences
        src_lens: lengths of input sequences
        tgt_ids: labels for the generator output
        tgt_lens: lengths of the generator targets
        gating_labels: labels for the gating head
        turn_domain: list of the domains

        """
        return {
            "src_ids": NeuralType(('B', 'T'), ChannelType()),
            "src_lens": NeuralType(tuple('B'), LengthsType()),
            "tgt_ids": NeuralType(('B', 'D', 'T'), LabelsType()),
            "tgt_lens": NeuralType(('B', 'D'), LengthsType()),
            "gating_labels": NeuralType(('B', 'D'), LabelsType()),
            "turn_domain": NeuralType(),
        }

    def __init__(
        self,
        data_dir,
        domains,
        all_domains,
        vocab,
        slots,
        gating_dict,
        num_samples=-1,
        batch_size=16,
        mode='train',
        dataset_type=MultiWOZDataset,
        shuffle=False,
        num_workers=0,
        input_dropout=0,
        is_training=False,
    ):

        dataset_params = {
            'data_dir': data_dir,
            'domains': domains,
            'num_samples': num_samples,
            'mode': mode,
            'shuffle': shuffle,
            'all_domains': all_domains,
            'vocab': vocab,
            'slots': slots,
            'gating_dict': gating_dict,
        }
        super().__init__(dataset_type, dataset_params, batch_size=batch_size)

        if self._placement == nemo.core.DeviceType.AllGpu:
            sampler = pt_data.distributed.DistributedSampler(self._dataset)
        else:
            sampler = None

        self._dataloader = pt_data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            shuffle=sampler is None,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            sampler=sampler,
        )
        self.pad_id = self._dataset.vocab.pad_id
        self.gating_dict = self._dataset.gating_dict
        self.input_dropout = input_dropout
        self.is_training = is_training
        self.vocab = self._dataset.vocab
        self.slots = self._dataset.slots

    def _collate_fn(self, data):
        """ data is a list of batch_size sample
        each sample is a dictionary of features
        """

        def pad_batch_context(sequences):
            '''
            merge from batch * sent_len to batch * max_len
            '''
            lengths = [len(seq) for seq in sequences]
            max_len = 1 if max(lengths) == 0 else max(lengths)
            for i, seq in enumerate(sequences):
                sequences[i] = seq + [1] * (max_len - len(seq))
            return torch.tensor(sequences), torch.tensor(lengths)

        def pad_batch_response(sequences, pad_id):
            '''
            merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
            '''
            lengths = []
            for bsz_seq in sequences:
                length = [len(v) for v in bsz_seq]
                lengths.append(length)
            max_len = max([max(l) for l in lengths])
            padded_seqs = []
            for bsz_seq in sequences:
                pad_seq = []
                for v in bsz_seq:
                    v = v + [pad_id] * (max_len - len(v))
                    pad_seq.append(v)
                padded_seqs.append(pad_seq)
            padded_seqs = torch.tensor(padded_seqs)
            lengths = torch.tensor(lengths)
            return padded_seqs, lengths

        data.sort(key=lambda x: len(x['context_ids']), reverse=True)
        item_info = {}
        for key in data[0]:
            item_info[key] = [item[key] for item in data]

        src_ids, src_lens = pad_batch_context(item_info['context_ids'])
        tgt_ids, tgt_lens = pad_batch_response(item_info['responses_ids'], self._dataset.vocab.pad_id)
        gating_label = torch.tensor(item_info['gating_label'])
        turn_domain = torch.tensor(item_info['turn_domain'])

        if self.input_dropout > 0 and self.is_training:
            bi_mask = np.random.binomial([np.ones(src_ids.size())], 1.0 - self.input_dropout)[0]
            rand_mask = torch.Tensor(bi_mask).long().to(src_ids.device)
            src_ids = src_ids * rand_mask

        return (
            src_ids.to(self._device),
            src_lens.to(self._device),
            tgt_ids.to(self._device),
            tgt_lens.to(self._device),
            gating_label.to(self._device),
            turn_domain.to(self._device),
        )

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self._dataloader
