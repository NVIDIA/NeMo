__all__ = ['DecoderRNN',
           'EncoderRNN']

import random

import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from torch import nn

from nemo.backends.pytorch.common.parts import Attention
from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_types import NeuralType, AxisType, BatchTag, TimeTag, \
    ChannelTag
from nemo.utils.misc import pad_to


class EncoderRNN(TrainableNM):
    """ Simple RNN-based encoder using GRU cells
    """

    @staticmethod
    def create_ports():
        input_ports = {
            'inputs': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            })
        }
        output_ports = {
            'outputs': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
            'hidden': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            })
        }

    def __init__(self,
                 input_dim,
                 emb_dim,
                 hid_dim,
                 dropout,
                 pad_idx=1,
                 n_layers=1,
                 embedding_to_load=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.emb = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        if embedding_to_load is not None:
            self.embedding.weight.data.copy_(embedding_to_load)
        else:
            self.embedding.weight.data.normal_(0, 0.1)
        self.rnn = nn.GRU(emb_dim,
                          hid_dim,
                          n_layers,
                          batch_first=True,
                          dropout=dropout,
                          bidirectional=True)

    def forward(self, inputs, input_lengths=None):
        embedded = self.emb(inputs)
        embedded = self.dropout(embedded)
        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded,
                                                         input_lengths,
                                                         batch_first=True)

        outputs, hidden = self.rnn(embedded)
        # outputs of shape (seq_len, batch, num_directions * hidden_size)
        # hidden of shape (num_layers * num_directions, batch, hidden_size)
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                          batch_first=True)
        else:
            outputs = outputs.transpose(0, 1)
        # outputs of shape: (batch, seq_len, num_directions * hidden_size)

        batch_size = hidden.size()[1]

        # separate final hidden states by layer and direction
        hidden = hidden.view(self.rnn.num_layers,
                             2 if self.rnn.bidirectional else 1,
                             batch_size,
                             self.rnn.hidden_size)
        hidden = hidden.tranpose(2, 0).tranpose(1, 2)
        hidden = hidden.reshape(batch_size, self.rnn.num_layers, -1)
        # hidden is now of shape (batch, seq_len, num_directions * hidden_size)
        self.to(self._device)
        return outputs, hidden


class DecoderRNN(TrainableNM):
    """Simple RNN-based decoder with attention.

    Args:
        input_dim (int): Total number of symbols to use.
                         This is usually the vocabulary size.
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

    @staticmethod
    def create_ports():
        input_ports = {
            'targets': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag)
            }),
            'encoder_outputs': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }, optional=True)
        }
        output_ports = {
            'log_probs': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(ChannelTag)
            }),
            'attention_weights': NeuralType({
                0: AxisType(BatchTag),
                1: AxisType(TimeTag),
                2: AxisType(TimeTag)
            }, optional=True)
        }
        return input_ports, output_ports

    def __init__(self,
                 input_dim,
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
                 **kwargs):
        super().__init__(**kwargs)

        self.bos_id = bos_id
        self.attention_type = attention_type
        self.teacher_forcing = teacher_forcing
        self.curriculum_learning = curriculum_learning
        self.rnn_type = rnn_type

        input_dim = pad_to(input_dim, 8)  # 8-divisors trick
        self.embedding = nn.Embedding(input_dim, hidden_size)
        # noinspection PyTypeChecker
        self.in_dropout = nn.Dropout(in_dropout)
        rnn_class = getattr(nn, rnn_type.upper())
        self.rnn = rnn_class(hidden_size, hidden_size, n_layers,
                             dropout=(0 if n_layers == 1 else gru_dropout),
                             batch_first=True)
        self.out = nn.Linear(hidden_size, input_dim)
        if tie_emb_out_weights:
            self.out.weight = self.embedding.weight  # Weight tying
        self.attention = Attention(hidden_size, attention_method,
                                   dropout=attn_dropout)

        # self.apply(init_weights)
        # self.gru.apply(init_weights)
        self.to(self._device)

    def forward(self, targets, encoder_outputs=None):
        if (not self.training) \
                or (random.random() <= self.teacher_forcing):  # Fast option
            # Removing last char (dont need to calculate loss) and add bos
            # noinspection PyTypeChecker
            decoder_inputs = F.pad(
                targets[:, :-1], (1, 0), value=self.bos_id
            )  # BT
            log_probs, _, attention_weights = \
                self.forward_step(decoder_inputs, encoder_outputs)
        else:
            log_probs, attention_weights = \
                self.forward_cl(targets, encoder_outputs)

        return log_probs, attention_weights

    def forward_step(self, decoder_inputs,
                     encoder_outputs=None, decoder_hidden=None):
        """(BT, BTC@?, hBC@?) -> (BTC, hBC, BTT@?)"""

        # Inputs
        decoder_inputs = self.embedding(decoder_inputs)
        # noinspection PyCallingNonCallable
        decoder_inputs = self.in_dropout(decoder_inputs)

        # RNN
        if self.rnn_type == 'gru' and decoder_hidden is not None:
            decoder_hidden = decoder_hidden[0]
        decoder_outputs, decoder_hidden = self.rnn(
            decoder_inputs, decoder_hidden
        )
        if self.rnn_type == 'gru':
            decoder_hidden = (decoder_hidden,)

        # Outputs
        attention_weights = None
        if self.attention_type == 'post':
            decoder_outputs, attention_weights = self.attention(
                decoder_outputs, encoder_outputs
            )
        decoder_outputs = self.out(decoder_outputs)

        # Log probs
        log_probs = F.log_softmax(decoder_outputs, dim=-1)

        return log_probs, decoder_hidden, attention_weights

    def forward_cl(self, targets, encoder_outputs=None):
        """(BT, BTC@?) -> (BTC, BTT@?)"""

        decoder_input = torch.empty(
            targets.size(0), 1,
            dtype=torch.long, device=self._device
        ).fill_(self.bos_id)
        decoder_hidden = None
        log_probs = []
        attention_weights = []

        max_len = targets.size(1)
        rands = torch.rand(max_len)  # Precalculate randomness
        for i in range(max_len):
            step_log_prob, decoder_hidden, step_attention_weights = \
                self.forward_step(
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
