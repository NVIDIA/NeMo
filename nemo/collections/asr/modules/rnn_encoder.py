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

import logging
import math
from collections import OrderedDict

import torch
import torch.distributed
import torch.nn as nn

from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling, StackingSubsampling
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType, SpectrogramType

__all__ = ['RNNEncoder']


class RNNEncoder(NeuralModule, Exportable):
    """
    The encoder for ASR model of Conformer.
    Based on this paper:
    'Conformer: Convolution-augmented Transformer for Speech Recognition' by Anmol Gulati et al.
    https://arxiv.org/abs/2005.08100

    Args:
        feat_in (int): the size of feature channels
        n_layers (int): number of layers of ConformerBlock
        d_model (int): the hidden size of the model
        feat_out (int): the size of the output features
            Defaults to -1 (means feat_out is d_model)
        subsampling (str): the method of subsampling, choices=['vggnet', 'striding']
            Defaults to striding.
        subsampling_factor (int): the subsampling factor which should be power of 2
            Defaults to 4.
        subsampling_conv_channels (int): the size of the convolutions in the subsampling module
            Defaults to -1 which would set it to d_model.
        ff_expansion_factor (int): the expansion factor in feed forward layers
            Defaults to 4.
        self_attention_model (str): type of the attention layer and positional encoding
            'rel_pos': relative positional embedding and Transformer-XL
            'abs_pos': absolute positional embedding and Transformer
            default is rel_pos.
        pos_emb_max_len (int): the maximum length of positional embeddings
            Defaulst to 5000
        n_heads (int): number of heads in multi-headed attention layers
            Defaults to 4.
        xscaling (bool): enables scaling the inputs to the multi-headed attention layers by sqrt(d_model)
            Defaults to True.
        untie_biases (bool): whether to not share (untie) the bias weights between layers of Transformer-XL
            Defaults to True.
        conv_kernel_size (int): the size of the convolutions in the convolutional modules
            Defaults to 31.
        dropout (float): the dropout rate used in all layers except the attention layers
            Defaults to 0.1.
        dropout_emb (float): the dropout rate used for the positional embeddings
            Defaults to 0.1.
        dropout_att (float): the dropout rate used for the attention layer
            Defaults to 0.0.
    """

    def input_example(self):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        input_example = torch.randn(16, self._feat_in, 256).to(next(self.parameters()).device)
        input_example_length = torch.randint(0, 256, (16,)).to(next(self.parameters()).device)
        return tuple([input_example, input_example_length])

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            }
        )

    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        proj_out=-1,
        rnn_type='lstm',
        subsampling='striding',
        subsampling_factor=4,
        subsampling_conv_channels=-1,
        dropout=0.1,
        bidirectional=True
    ):
        super().__init__()

        self.d_model = d_model
        self._feat_in = feat_in

        if subsampling_conv_channels == -1:
            subsampling_conv_channels = proj_out
        if subsampling and subsampling_factor > 1:
            if subsampling == 'stacking':
                self.pre_encode = StackingSubsampling(subsampling_factor=subsampling_factor, feat_in=feat_in, feat_out=proj_out)
            else:
                self.pre_encode = ConvSubsampling(
                    subsampling=subsampling,
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=proj_out,
                    conv_channels=subsampling_conv_channels,
                    activation=nn.ReLU(),
                )
        else:
            self.pre_encode = nn.Linear(feat_in, proj_out)

        self._feat_out = proj_out

        self.layers = nn.ModuleList()

        SUPPORTED_RNN = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}
        if rnn_type not in SUPPORTED_RNN:
            raise ValueError(f"rnn_type can be one from the following:{SUPPORTED_RNN.keys()}")
        else:
            rnn_module = SUPPORTED_RNN[rnn_type]

        for i in range(n_layers):
            rnn_proj_out = proj_out//2 if bidirectional else proj_out
            if rnn
            if rnn_type == "lstm":
                layer = rnn_module(
                    input_size=self._feat_out,
                    hidden_size=d_model,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                    proj_size=rnn_proj_out
                )
            self.layers.append(layer)
            self.layers.append(nn.LayerNorm(proj_out))
            self.layers.append(nn.Dropout(p=dropout))
            self._feat_out = proj_out

        if proj_out > 0 and self._feat_out != proj_out:
            self.out_proj = nn.Linear(self._feat_out, proj_out)
            self._feat_out = proj_out
        else:
            self.out_proj = None

    @typecheck()
    def forward(self, audio_signal, length=None):
        max_audio_length: int = audio_signal.size(-1)

        if length is None:
            length = audio_signal.new_full(
                audio_signal.size(0), max_audio_length, dtype=torch.int32, device=self.seq_range.device
            )

        audio_signal = torch.transpose(audio_signal, 1, 2)

        if isinstance(self.pre_encode, ConvSubsampling) or isinstance(self.pre_encode, StackingSubsampling):
            audio_signal, length = self.pre_encode(audio_signal, length)
        else:
            audio_signal = self.pre_encode(audio_signal)

        for lth, layer in enumerate(self.layers):
            if isinstance(layer, torch.nn.Dropout) or isinstance(layer, torch.nn.LayerNorm):
                audio_signal = layer(audio_signal)
            else:
                audio_signal, _ = layer(audio_signal)

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        audio_signal = torch.transpose(audio_signal, 1, 2)
        return audio_signal, length
