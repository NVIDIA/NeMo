""" These NeuralModules are based on PyTorch's tutorial:
https://pytorch.org/tutorials/beginner/chatbot_tutorial.html
"""
import random
from typing import Iterable, Mapping, Optional

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from .....core import DeviceType
from .....core.neural_types import *
from ...nm import DataLayerNM, LossNM, TrainableNM
from ..chatbot import data
from nemo.utils.decorators import add_port_docs


class DialogDataLayer(DataLayerNM):
    """Class representing data layer for a chatbot."""

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            "src": NeuralType(('T', 'B'), ChannelType()),
            "src_lengths": NeuralType(tuple('B'), LengthsType()),
            "tgt": NeuralType(('T', 'B'), LabelsType()),
            "mask": NeuralType(('T', 'B'), ChannelType()),
            "max_tgt_lengths": NeuralType(axes=None),
        }

    def __init__(self, batch_size, corpus_name, datafile, min_count=3):
        super().__init__()

        self._batch_size = batch_size
        self._corpus_name = corpus_name
        self._datafile = datafile
        self._min_count = min_count
        voc, pairs = data.loadPrepareData(self._corpus_name, self._datafile)
        self.voc = voc
        self.pairs = data.trimRareWords(voc, pairs, self._min_count)

        self._device = t.device("cuda" if self.placement == DeviceType.GPU else "cpu")
        self._dataloader = []
        for i in range(self.__len__()):
            self._dataloader.append(self.__getitem__(i))

    def __len__(self):
        return len(self.pairs) // self._batch_size

    def __getitem__(self, idx):
        return [
            x.to(self._device) if isinstance(x, t.Tensor) else x
            for x in data.batch2TrainData(self.voc, [random.choice(self.pairs) for _ in range(self._batch_size)],)
        ]

    def get_weights(self) -> Iterable[Optional[Mapping]]:
        return None

    @property
    def data_iterator(self):
        return self._dataloader

    @property
    def dataset(self):
        return None


class EncoderRNN(TrainableNM):
    """RNN-based encoder Neural Module
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "input_seq": NeuralType(('T', 'B'), ChannelType()),
            "input_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            "outputs": NeuralType(('T', 'B', 'D'), ChannelType()),
            "hidden": NeuralType(('B', 'D'), ChannelType()),
        }

    def __init__(self, voc_size, encoder_n_layers, hidden_size, dropout, bidirectional=True):
        super().__init__()

        self.voc_size = voc_size
        self.n_layers = encoder_n_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.bidirectional = bidirectional

        # Define layers
        self.embedding = nn.Embedding(self.voc_size, self.hidden_size)
        self.embedding_dropout = nn.Dropout(self.dropout)
        # Initialize GRU; the input_size and hidden_size params are both set
        # to 'hidden_size'
        #   because our input size is a word embedding with number of
        #   features == hidden_size
        self.gru = nn.GRU(
            self.hidden_size,
            self.hidden_size,
            self.n_layers,
            dropout=(0 if self.n_layers == 1 else self.dropout),
            bidirectional=self.bidirectional,
        )
        self._device = t.device("cuda" if self.placement == DeviceType.GPU else "cpu")
        self.to(self._device)

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        # Pack padded batch of sequences for RNN module
        packed = t.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = t.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]
        # Return output and final hidden state
        return outputs, hidden


class LuongAttnDecoderRNN(TrainableNM):
    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "targets": NeuralType(('T', 'B'), LabelsType()),
            "encoder_outputs": NeuralType(('T', 'B', 'D'), ChannelType()),
            "max_target_len": NeuralType(axes=None),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        outputs:
            0: AxisType(TimeTag)

            1: AxisType(BatchTag)

            2: AxisType(ChannelTag)

        hidden:
            0: AxisType(BatchTag)

            1: AxisType(ChannelTag)
        """
        return {
            "outputs": NeuralType(('T', 'B', 'D'), ChannelType()),
            "hidden": NeuralType(('B', 'D'), ChannelType()),
        }

    def __init__(self, attn_model, hidden_size, voc_size, decoder_n_layers, dropout):
        super().__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.voc_size = voc_size
        # output_size
        self.output_size = voc_size
        self.n_layers = decoder_n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(self.voc_size, self.hidden_size)
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(
            self.hidden_size, self.hidden_size, self.n_layers, dropout=(0 if self.n_layers == 1 else self.dropout),
        )
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # Luong attention layer
        class Attn(t.nn.Module):
            def __init__(self, method, hidden_size):
                super(Attn, self).__init__()
                self.method = method
                if self.method not in ["dot", "general", "concat"]:
                    raise ValueError(self.method, "is not an appropriate attention method.")
                self.hidden_size = hidden_size
                if self.method == "general":
                    self.attn = t.nn.Linear(self.hidden_size, hidden_size)
                elif self.method == "concat":
                    self.attn = t.nn.Linear(self.hidden_size * 2, hidden_size)
                    self.v = t.nn.Parameter(t.FloatTensor(hidden_size))

            def dot_score(self, hidden, encoder_output):
                return t.sum(hidden * encoder_output, dim=2)

            def general_score(self, hidden, encoder_output):
                energy = self.attn(encoder_output)
                return t.sum(hidden * energy, dim=2)

            def concat_score(self, hidden, encoder_output):
                energy = self.attn(t.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output,), 2,)).tanh()
                return t.sum(self.v * energy, dim=2)

            def forward(self, hidden, encoder_outputs):
                # Calculate the attention weights (energies) based on the
                # given method
                if self.method == "general":
                    attn_energies = self.general_score(hidden, encoder_outputs)
                elif self.method == "concat":
                    attn_energies = self.concat_score(hidden, encoder_outputs)
                elif self.method == "dot":
                    attn_energies = self.dot_score(hidden, encoder_outputs)

                # Transpose max_length and batch_size dimensions
                attn_energies = attn_energies.t()

                # Return the softmax normalized probability scores (with
                # added dimension)
                return F.softmax(attn_energies, dim=1).unsqueeze(1)

        self.attn = Attn(self.attn_model, self.hidden_size)

        self._device = t.device("cuda" if self.placement == DeviceType.GPU else "cpu")
        self.to(self._device)

    def one_step_forward(self, embedded, last_hidden, encoder_outputs):
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted
        # sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = t.cat((rnn_output, context), 1)
        concat_output = t.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden

    def forward(self, targets, encoder_outputs, max_target_len):
        SOS_token = 1  # Start-of-sentence token

        decoder_input = t.LongTensor([[SOS_token for _ in range(encoder_outputs.shape[1])]])
        decoder_input = decoder_input.to(self._device)
        decoder_hidden = None
        decoder_output = []
        for step_t in range(max_target_len):
            decoder_inpt_embd = self.embedding(decoder_input)
            decoder_step_output, decoder_hidden = self.one_step_forward(
                embedded=decoder_inpt_embd, last_hidden=decoder_hidden, encoder_outputs=encoder_outputs,
            )
            decoder_output.append(decoder_step_output)
            # Teacher forcing: next input is current target
            decoder_input = targets[step_t].view(1, -1)
        result = t.stack(decoder_output, dim=0)
        return result, decoder_hidden


class MaskedXEntropyLoss(LossNM):
    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "predictions": NeuralType(('T', 'B', 'D'), ChannelType()),
            "target": NeuralType(('T', 'B'), LabelsType()),
            "mask": NeuralType(('T', 'B'), ChannelType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(axes=None, elements_type=LossType())}

    def __init__(self):
        super().__init__()

        self._device = t.device("cuda" if self.placement == DeviceType.GPU else "cpu")

    def _loss(self, inp, target, mask):
        inp = inp.view(-1, inp.shape[2])
        mask = mask.view(-1, 1)
        crossEntropy = -t.log(t.gather(inp, 1, target.view(-1, 1)))
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(self._device)
        return loss

    def _loss_function(self, **kwargs):
        return self._loss(*(kwargs.values()))


class GreedyLuongAttnDecoderRNN(TrainableNM):
    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {"encoder_outputs": NeuralType(('T', 'B', 'D'), ChannelType())}

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            "outputs": NeuralType(('T', 'B'), ChannelType()),
            "hidden": NeuralType(('B', 'D'), ChannelType()),
        }

    def __init__(self, attn_model, hidden_size, voc_size, decoder_n_layers, dropout, max_dec_steps=10):
        super().__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.voc_size = voc_size
        self.output_size = voc_size
        self.n_layers = decoder_n_layers
        self.dropout = dropout
        self.max_decoder_steps = max_dec_steps

        # Define layers
        self.embedding = nn.Embedding(self.voc_size, self.hidden_size)
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(
            self.hidden_size, self.hidden_size, self.n_layers, dropout=(0 if self.n_layers == 1 else self.dropout),
        )
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # Luong attention layer
        class Attn(t.nn.Module):
            def __init__(self, method, hidden_size):
                super(Attn, self).__init__()
                self.method = method
                if self.method not in ["dot", "general", "concat"]:
                    raise ValueError(self.method, "is not an appropriate attention method.")
                self.hidden_size = hidden_size
                if self.method == "general":
                    self.attn = t.nn.Linear(self.hidden_size, hidden_size)
                elif self.method == "concat":
                    self.attn = t.nn.Linear(self.hidden_size * 2, hidden_size)
                    self.v = t.nn.Parameter(t.FloatTensor(hidden_size))

            def dot_score(self, hidden, encoder_output):
                return t.sum(hidden * encoder_output, dim=2)

            def general_score(self, hidden, encoder_output):
                energy = self.attn(encoder_output)
                return t.sum(hidden * energy, dim=2)

            def concat_score(self, hidden, encoder_output):
                energy = self.attn(t.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output,), 2,)).tanh()
                return t.sum(self.v * energy, dim=2)

            def forward(self, hidden, encoder_outputs):
                # Calculate the attention weights (energies) based on the
                # given method
                if self.method == "general":
                    attn_energies = self.general_score(hidden, encoder_outputs)
                elif self.method == "concat":
                    attn_energies = self.concat_score(hidden, encoder_outputs)
                elif self.method == "dot":
                    attn_energies = self.dot_score(hidden, encoder_outputs)

                # Transpose max_length and batch_size dimensions
                attn_energies = attn_energies.t()

                # Return the softmax normalized probability scores (with
                # added dimension)
                return F.softmax(attn_energies, dim=1).unsqueeze(1)

        self.attn = Attn(self.attn_model, self.hidden_size)

        self._device = t.device("cuda" if self.placement == DeviceType.GPU else "cpu")
        self.to(self._device)

    def one_step_forward(self, embedded, last_hidden, encoder_outputs):
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted
        # sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = t.cat((rnn_output, context), 1)
        concat_output = t.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden

    def forward(self, encoder_outputs):
        SOS_token = 1  # Start-of-sentence token
        encoder_outputs = encoder_outputs.detach()

        decoder_input = t.LongTensor([[SOS_token for _ in range(encoder_outputs.shape[1])]])
        decoder_input = decoder_input.to(self._device)
        decoder_hidden = None
        decoder_output = []
        done = False
        for step_t in range(self.max_decoder_steps):
            decoder_inpt_embd = self.embedding(decoder_input)
            decoder_step_output, decoder_hidden = self.one_step_forward(
                embedded=decoder_inpt_embd, last_hidden=decoder_hidden, encoder_outputs=encoder_outputs,
            )
            decoder_output.append(decoder_step_output)
            # Teacher forcing: next input is current target
            _, topi = decoder_step_output.topk(1)
            topi = topi.detach()
            # if topi.item() == EOS_token:
            #  break
            decoder_input = t.LongTensor([[topi[i][0] for i in range(topi.shape[0])]])
            decoder_input = decoder_input.to(self._device)
            # decoder_input = targets[step_t].view(1, -1)
        result_logits = t.stack(decoder_output, dim=0)
        _, result = result_logits.topk(1)
        return result.squeeze(-1), decoder_hidden
        # return result, decoder_hidden
