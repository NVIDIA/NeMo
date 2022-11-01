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

import math
from collections import OrderedDict

import torch
import torch.distributed
import torch.nn as nn

from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import EncodedRepresentation, NeuralType, SpectrogramType
from nemo.core.neural_types.elements import MaskType

__all__ = ['SpeakerBeam']


speakerbeam_activations = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
}


class SpeakerBeam(NeuralModule, Exportable):
    """
    SpeakerBeam
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8736286

    Args:
        feat_in (int): the size of feature channels
        n_layers (int): number of layers of ConformerBlock
        d_model (int): the hidden size of the model
        feat_out (int): the size of the output features
            Defaults to -1 (means feat_out is d_model)
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "features": NeuralType(('B', 'D'), EncodedRepresentation()),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return OrderedDict({"audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),})

    def __init__(
        self, feat_in, n_layers, d_model, activation, feat_in_adapt, n_layers_adapt, d_models_adapt, activation_adapt, lstm_layers=1
    ):
        super().__init__()

        self.d_model = d_model
        self._feat_in = feat_in
        self._activation = activation

        self._feat_in_adapt = feat_in_adapt
        self.layers_adapt = n_layers_adapt
        self.d_model_adapt = d_models_adapt
        self._activation_adapt = activation_adapt

        self.layers = nn.ModuleList()
        hidden_size = feat_in
        for i in range(n_layers):
            sub_lstm_layer = nn.LSTM(
                input_size=hidden_size, hidden_size=self.d_model, num_layers=lstm_layers, batch_first=True, bidirectional=True,
            )
            sub_linear = nn.Linear(self.d_model * 2, self.d_model, bias=True)
            sub_activation = speakerbeam_activations[self._activation]()
            self.layers.append(sub_lstm_layer)
            seq = nn.Sequential(sub_linear, sub_activation)
            hidden_size = self.d_model
            self.layers.append(seq)

        sub_linear = nn.Linear(self.d_model, self._feat_in, bias=True)
        sub_activation = speakerbeam_activations["sigmoid"]()
        self.layers.append(nn.Sequential(sub_linear, sub_activation))

        self.layers_adapt = nn.ModuleList()
        hidden_size = self._feat_in_adapt
        for i in range(n_layers_adapt):
            seq = nn.Sequential(
                nn.Linear(hidden_size, self.d_model_adapt), speakerbeam_activations[self._activation_adapt]()
            )
            hidden_size = self.d_model_adapt
            self.layers_adapt.append(seq)

        self.layers_adapt.append(nn.Linear(hidden_size, self.d_model))

    @typecheck()
    def forward(self, audio_signal, features):
        for lth, layer in enumerate(self.layers_adapt):
            features = layer(features)
        features = features.unsqueeze(1)
        audio_signal = audio_signal.transpose(1, 2)
        for lth, layer in enumerate(self.layers):
            audio_signal = layer(audio_signal)

            if isinstance(audio_signal, tuple):
                audio_signal, _ = audio_signal
            if lth == 1:
                audio_signal = audio_signal * features

        audio_signal = audio_signal.transpose(1, 2)
        return audio_signal
