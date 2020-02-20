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

import random

import torch
import torch.nn.functional as pt_f
from torch import nn

from nemo.backends.pytorch.common.parts import Attention
from nemo.backends.pytorch.nm import TrainableNM
from nemo.core import *
from nemo.utils.decorators import add_port_docs
from nemo.utils.misc import pad_to

__all__ = ['DecoderRNN', 'EncoderRNN']


class DecoderRNN(TrainableNM):
    """Simple RNN-based decoder with attention.

    Args:
        voc_size (int): Total number of symbols to use
        bos_id (int): Label position of start of string symbol
        hidden_size (int): Size of hidden vector to use in RNN
        attention_method (str): Method of using attention to pass in
            `Attention` constructor.
            Defaults to 'general'.
        attention_type (str): String type of attention describing time to apply
            attention. Could be on of ['post', 'none'].
            Defaults to 'post'.
        in_dropout (float): Float value of embedding dropout.
            Defaults to 0.2.
        gru_dropout (float): Float value of RNN interlayers dropout
            Defaults to 0.2.
        attn_dropout (float): Float value of attention dropout to pass to
            `Attention` constructor
            Defaults to 0.0.
        teacher_forcing (float): Probability of applying full teacher forcing
            method at each step.
            Defaults to 1.0.
        curriculum_learning (float): If teacher forcing is not applying, this
            value indicates probability of using target token from next step.
            Defaults to 0.5.
        rnn_type (str): Type of RNN to use. Could be one of ['gru', 'lstm'].
            Defaults to 'gru'.
        n_layers (int): Number of layers to use in RNN.
            Defaults to 2.
        tie_emb_out_weights (bool): Whether to tie embedding and output
            weights.
            Defaults to True.

    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            'targets': NeuralType(('B', 'T'), LabelsType()),
            'encoder_outputs': NeuralType(('B', 'T', 'D'), ChannelType(), True),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            'log_probs': NeuralType(('B', 'T', 'D'), LogprobsType()),
            'attention_weights': NeuralType(('B', 'T', 'T'), ChannelType(), True),
        }

    def __init__(
        self,
        voc_size,
        bos_id,
        hidden_size,
        attention_method='general',
        attention_type='post',
        in_dropout=0.2,
        gru_dropout=0.2,
        attn_dropout=0.0,
        teacher_forcing=1.0,
        curriculum_learning=0.5,
        rnn_type='gru',
        n_layers=2,
        tie_emb_out_weights=True,
    ):
        super().__init__()

        self.bos_id = bos_id
        self.attention_type = attention_type
        self.teacher_forcing = teacher_forcing
        self.curriculum_learning = curriculum_learning
        self.rnn_type = rnn_type

        voc_size = pad_to(voc_size, 8)  # 8-divisors trick
        self.embedding = nn.Embedding(voc_size, hidden_size)
        # noinspection PyTypeChecker
        self.in_dropout = nn.Dropout(in_dropout)
        rnn_class = getattr(nn, rnn_type.upper())
        self.rnn = rnn_class(
            hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else gru_dropout), batch_first=True,
        )
        self.out = nn.Linear(hidden_size, voc_size)
        if tie_emb_out_weights:
            self.out.weight = nn.Parameter(self.embedding.weight)  # Weight tying
        self.attention = Attention(hidden_size, attention_method, dropout=attn_dropout)

        # self.apply(init_weights)
        # self.gru.apply(init_weights)
        self.to(self._device)

    def forward(self, targets, encoder_outputs=None):
        if (not self.training) or (random.random() <= self.teacher_forcing):  # Fast option
            # Removing last char (dont need to calculate loss) and add bos
            # noinspection PyTypeChecker
            decoder_inputs = pt_f.pad(targets[:, :-1], (1, 0), value=self.bos_id)  # BT
            log_probs, _, attention_weights = self.forward_step(decoder_inputs, encoder_outputs)
        else:
            log_probs, attention_weights = self.forward_cl(targets, encoder_outputs)

        return log_probs, attention_weights

    def forward_step(self, decoder_inputs, encoder_outputs=None, decoder_hidden=None):
        """(BT, BTC@?, hBC@?) -> (BTC, hBC, BTT@?)"""

        # Inputs
        decoder_inputs = self.embedding(decoder_inputs)
        # noinspection PyCallingNonCallable
        decoder_inputs = self.in_dropout(decoder_inputs)

        # RNN
        if self.rnn_type == 'gru' and decoder_hidden is not None:
            decoder_hidden = decoder_hidden[0]
        decoder_outputs, decoder_hidden = self.rnn(decoder_inputs, decoder_hidden)
        if self.rnn_type == 'gru':
            decoder_hidden = (decoder_hidden,)

        # Outputs
        attention_weights = None
        if self.attention_type == 'post':
            decoder_outputs, attention_weights = self.attention(decoder_outputs, encoder_outputs)
        decoder_outputs = self.out(decoder_outputs)

        # Log probs
        log_probs = pt_f.log_softmax(decoder_outputs, dim=-1)

        return log_probs, decoder_hidden, attention_weights

    def forward_cl(self, targets, encoder_outputs=None):
        """(BT, BTC@?) -> (BTC, BTT@?)"""

        decoder_input = torch.empty(targets.size(0), 1, dtype=torch.long, device=self._device).fill_(self.bos_id)
        decoder_hidden = None
        log_probs = []
        attention_weights = []

        max_len = targets.size(1)
        rands = torch.rand(max_len)  # Precalculate randomness
        for i in range(max_len):
            (step_log_prob, decoder_hidden, step_attention_weights,) = self.forward_step(
                decoder_input, encoder_outputs, decoder_hidden
            )
            log_probs.append(step_log_prob)
            attention_weights.append(step_attention_weights)

            if rands[i] <= self.curriculum_learning:
                decoder_input = targets[:, i].view(-1, 1).detach()
            else:
                decoder_input = step_log_prob.argmax(-1).detach()

        log_probs = torch.cat(log_probs, dim=1)
        if len(attention_weights) and attention_weights[0] is not None:
            attention_weights = torch.cat(attention_weights, dim=1)
        else:
            attention_weights = None

        return log_probs, attention_weights


class EncoderRNN(TrainableNM):
    """ Simple RNN-based encoder using GRU cells """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            'inputs': NeuralType(('B', 'T'), ChannelType()),
            'input_lens': NeuralType(tuple('B'), LengthsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            'outputs': NeuralType(('B', 'T', 'D'), ChannelType()),
            'hidden': NeuralType(('B', 'T', 'D'), ChannelType()),
        }

    def __init__(
        self, input_dim, emb_dim, hid_dim, dropout, n_layers=1, pad_idx=1, embedding_to_load=None, sum_hidden=True
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        if embedding_to_load is not None:
            self.embedding.weight.data.copy_(embedding_to_load)
        else:
            self.embedding.weight.data.normal_(0, 0.1)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.sum_hidden = sum_hidden
        self.to(self._device)

    def forward(self, inputs, input_lens=None):
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)
        if input_lens is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lens, batch_first=True)

        outputs, hidden = self.rnn(embedded)
        # outputs of shape (seq_len, batch, num_directions * hidden_size)
        # hidden of shape (num_layers * num_directions, batch, hidden_size)
        if input_lens is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs = outputs.transpose(0, 1)
        # outputs of shape: (batch, seq_len, num_directions * hidden_size)

        batch_size = hidden.size()[1]

        # separate final hidden states by layer and direction
        hidden = hidden.view(self.rnn.num_layers, 2 if self.rnn.bidirectional else 1, batch_size, self.rnn.hidden_size)
        hidden = hidden.transpose(2, 0).transpose(1, 2)
        # hidden shape: batch x num_layer x num_directions x hidden_size
        if self.sum_hidden and self.rnn.bidirectional:
            hidden = hidden[:, :, 0, :] + hidden[:, :, 1, :]
            outputs = outputs[:, :, : self.rnn.hidden_size] + outputs[:, :, self.rnn.hidden_size :]
        else:
            hidden = hidden.reshape(batch_size, self.rnn.num_layers, -1)
        # hidden is now of shape (batch, num_layer, [num_directions] * hidden_size)

        return outputs, hidden
