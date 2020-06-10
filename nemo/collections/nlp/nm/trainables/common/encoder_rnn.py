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

from torch import nn

from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_types import *
from nemo.utils.decorators import add_port_docs

__all__ = ['EncoderRNN']


class EncoderRNN(TrainableNM):
    """ Simple RNN-based encoder using GRU cells - with input/output port definitions used in TRADE. """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            'inputs': NeuralType(('B', 'T'), TokenIndex()),
            'input_lens': NeuralType(tuple('B'), Length()),
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
