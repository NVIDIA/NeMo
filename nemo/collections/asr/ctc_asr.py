# -*- coding: utf-8 -*-

# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
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

from nemo.collections.asr.parts.jasper import JasperBlock, init_weights, jasper_activations
from nemo.core import AcousticEncodedRepresentation, LengthsType, LogprobsType, NeuralType, SpectrogramType
from nemo.core.apis import NeuralModuleAPI

__all__ = ['ConvASRDecoder', 'ConvASREncoder']


class ConvASREncoder(torch.nn.Module, NeuralModuleAPI):
    @property
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        jasper,
        activation,
        feat_in,
        normalization_mode="batch",
        residual_mode="add",
        norm_groups=-1,
        conv_mask=True,
        frame_splicing=1,
        init_mode='xavier_uniform',
    ):
        super(ConvASREncoder, self).__init__()
        activation = jasper_activations[activation]()
        feat_in = feat_in * frame_splicing

        self.__feat_in = feat_in

        residual_panes = []
        encoder_layers = []
        self.dense_residual = False
        for lcfg in jasper:
            dense_res = []
            if lcfg.get('residual_dense', False):
                residual_panes.append(feat_in)
                dense_res = residual_panes
                self.dense_residual = True
            groups = lcfg.get('groups', 1)
            separable = lcfg.get('separable', False)
            heads = lcfg.get('heads', -1)
            residual_mode = lcfg.get('residual_mode', residual_mode)
            se = lcfg.get('se', False)
            se_reduction_ratio = lcfg.get('se_reduction_ratio', 8)
            se_context_window = lcfg.get('se_context_window', -1)
            se_interpolation_mode = lcfg.get('se_interpolation_mode', 'nearest')
            kernel_size_factor = lcfg.get('kernel_size_factor', 1.0)
            stride_last = lcfg.get('stride_last', False)
            encoder_layers.append(
                JasperBlock(
                    feat_in,
                    lcfg['filters'],
                    repeat=lcfg['repeat'],
                    kernel_size=lcfg['kernel'],
                    stride=lcfg['stride'],
                    dilation=lcfg['dilation'],
                    dropout=lcfg['dropout'],
                    residual=lcfg['residual'],
                    groups=groups,
                    separable=separable,
                    heads=heads,
                    residual_mode=residual_mode,
                    normalization=normalization_mode,
                    norm_groups=norm_groups,
                    activation=activation,
                    residual_panes=dense_res,
                    conv_mask=conv_mask,
                    se=se,
                    se_reduction_ratio=se_reduction_ratio,
                    se_context_window=se_context_window,
                    se_interpolation_mode=se_interpolation_mode,
                    kernel_size_factor=kernel_size_factor,
                    stride_last=stride_last,
                )
            )
            feat_in = lcfg['filters']

        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, audio_signal, length=None):
        s_input, length = self.encoder(([audio_signal], length))
        if length is None:
            return s_input[-1]

        return s_input[-1], length


class ConvASRDecoder(torch.nn.Module, NeuralModuleAPI):
    @property
    def input_types(self):
        return {"encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation())}

    @property
    def output_types(self):
        return {"logprobs": NeuralType(('B', 'T', 'D'), LogprobsType())}

    def __init__(self, feat_in, num_classes, init_mode="xavier_uniform", vocabulary=None):
        super(ConvASRDecoder, self).__init__()

        if vocabulary is not None:
            if num_classes != len(vocabulary):
                raise ValueError(
                    f"If vocabulary is specified, it's length should be equal to the num_classes. But I got: num_classes={num_classes} and len(vocabluary)={len(vocabulary)}"
                )
            self.__vocabulary = vocabulary
        self._feat_in = feat_in
        # Add 1 for blank char
        self._num_classes = num_classes + 1

        self.decoder_layers = torch.nn.Sequential(
            torch.nn.Conv1d(self._feat_in, self._num_classes, kernel_size=1, bias=True)
        )
        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, encoder_output):
        return torch.nn.functional.log_softmax(self.decoder_layers(encoder_output).transpose(1, 2), dim=-1)
