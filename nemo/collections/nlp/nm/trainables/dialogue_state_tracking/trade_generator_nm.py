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


import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_types import *
from nemo.utils.decorators import add_port_docs

__all__ = ['TRADEGenerator']


class TRADEGenerator(TrainableNM):
    """
    The generator module for state tracking model TRADE
    Args:
        vocab (Vocab): an instance of Vocab containing the vocabularey
        embeddings (Tensor): word embedding matrix
        hid_size (int): hidden size of the GRU decoder
        dropout (float): dropout of the GRU
        slots (list): list of slots
        nb_gate (int): number of gates
        teacher_forcing (float): 0.5
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            'encoder_hidden': NeuralType(('B', 'T', 'D'), ChannelType()),
            'encoder_outputs': NeuralType(('B', 'T', 'D'), ChannelType()),
            'dialog_ids': NeuralType(('B', 'T'), elements_type=TokenIndex()),
            'dialog_lens': NeuralType(tuple('B'), elements_type=Length()),
            'targets': NeuralType(('B', 'D', 'T'), LabelsType(), optional=True),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        point_outputs: outputs of the generator
        gate_outputs: outputs of gating heads
        """
        return {
            'point_outputs': NeuralType(('B', 'T', 'D', 'D'), LogitsType()),
            'gate_outputs': NeuralType(('B', 'D', 'D'), LogitsType()),
        }

    def __init__(self, vocab, embeddings, hid_size, dropout, slots, nb_gate, teacher_forcing=0.5, max_res_len=10):
        super().__init__()
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.embedding = embeddings
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(hid_size, hid_size, dropout=dropout, batch_first=True)
        self.nb_gate = nb_gate
        self.hidden_size = hid_size
        self.w_ratio = nn.Linear(3 * hid_size, 1)
        self.w_gate = nn.Linear(hid_size, nb_gate)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots
        self.teacher_forcing = teacher_forcing
        # max_res_len is used in evaluation mode or when targets are not provided
        self.max_res_len = max_res_len

        self._slots_split_to_index()
        self.slot_emb = nn.Embedding(len(self.slot_w2i), hid_size)
        self.slot_emb.weight.data.normal_(0, 0.1)
        self.to(self._device)

    def _slots_split_to_index(self):
        split_slots = [slot.split('-') for slot in self.slots]
        domains = [split_slot[0] for split_slot in split_slots]
        slots = [split_slot[1] for split_slot in split_slots]
        split_slots = list({s: 0 for s in sum(split_slots, [])})
        self.slot_w2i = {split_slots[i]: i for i in range(len(split_slots))}
        self.domain_idx = torch.tensor([self.slot_w2i[domain] for domain in domains], device=self._device)
        self.subslot_idx = torch.tensor([self.slot_w2i[slot] for slot in slots], device=self._device)

    def forward(self, encoder_hidden, encoder_outputs, dialog_ids, dialog_lens, targets=None):
        if (not self.training) or (random.random() > self.teacher_forcing):
            use_teacher_forcing = False
        else:
            use_teacher_forcing = True

        batch_size = encoder_hidden.shape[0]

        if isinstance(targets, torch.Tensor):
            max_res_len = targets.shape[2]
            targets = targets.transpose(0, 1)
        else:
            max_res_len = self.max_res_len

        all_point_outputs = torch.zeros(len(self.slots), batch_size, max_res_len, self.vocab_size, device=self._device)
        all_gate_outputs = torch.zeros(len(self.slots), batch_size, self.nb_gate, device=self._device)

        domain_emb = self.slot_emb(self.domain_idx).to(self._device)
        subslot_emb = self.slot_emb(self.subslot_idx).to(self._device)
        slot_emb = domain_emb + subslot_emb
        slot_emb = slot_emb.unsqueeze(1)
        slot_emb = slot_emb.repeat(1, batch_size, 1)
        decoder_input = self.dropout(slot_emb).view(-1, self.hidden_size)
        hidden = encoder_hidden[:, 0:1, :].transpose(0, 1).repeat(len(self.slots), 1, 1)

        hidden = hidden.view(-1, self.hidden_size).unsqueeze(0)

        enc_len = dialog_lens.repeat(len(self.slots))

        maxlen = encoder_outputs.size(1)
        padding_mask_bool = ~(torch.arange(maxlen, device=self._device)[None, :] <= enc_len[:, None])
        padding_mask = torch.zeros_like(padding_mask_bool, dtype=encoder_outputs.dtype, device=self._device)
        padding_mask.masked_fill_(mask=padding_mask_bool, value=-np.inf)

        for wi in range(max_res_len):
            dec_state, hidden = self.rnn(decoder_input.unsqueeze(1), hidden)

            enc_out = encoder_outputs.repeat(len(self.slots), 1, 1)
            context_vec, logits, prob = TRADEGenerator.attend(enc_out, hidden.squeeze(0), padding_mask)

            if wi == 0:
                all_gate_outputs = torch.reshape(self.w_gate(context_vec), all_gate_outputs.size())

            p_vocab = TRADEGenerator.attend_vocab(self.embedding.weight, hidden.squeeze(0))
            p_gen_vec = torch.cat([dec_state.squeeze(1), context_vec, decoder_input], -1)
            vocab_pointer_switches = self.sigmoid(self.w_ratio(p_gen_vec))
            p_context_ptr = torch.zeros(p_vocab.size(), device=self._device)

            p_context_ptr.scatter_add_(1, dialog_ids.repeat(len(self.slots), 1), prob)

            final_p_vocab = (1 - vocab_pointer_switches).expand_as(
                p_context_ptr
            ) * p_context_ptr + vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
            pred_word = torch.argmax(final_p_vocab, dim=1)

            all_point_outputs[:, :, wi, :] = torch.reshape(
                final_p_vocab, (len(self.slots), batch_size, self.vocab_size)
            )

            if use_teacher_forcing and isinstance(targets, torch.Tensor):
                decoder_input = self.embedding(torch.flatten(targets[:, :, wi]))
            else:
                decoder_input = self.embedding(pred_word)

            decoder_input = decoder_input.to(self._device)
        all_point_outputs = all_point_outputs.transpose(0, 1).contiguous()
        all_gate_outputs = all_gate_outputs.transpose(0, 1).contiguous()
        return all_point_outputs, all_gate_outputs

    @staticmethod
    def attend(seq, cond, padding_mask):
        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)
        scores_ = scores_ + padding_mask
        scores = F.softmax(scores_, dim=1)
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, scores_, scores

    @staticmethod
    def attend_vocab(seq, cond):
        scores_ = cond.matmul(seq.transpose(1, 0))
        scores = F.softmax(scores_, dim=1)
        return scores
