# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
import copy
from collections import OrderedDict
from math import pi

import torch

__all__ = ["PerceiverEncoder"]

from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer

from nemo.core import NeuralModule, Exportable, typecheck
from nemo.core.neural_types import NeuralType, SpectrogramType, LengthsType, AcousticEncodedRepresentation


def fourier_encode(x, max_freq, num_bands=4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


class PerceiverEncoder(NeuralModule, Exportable):
    def __init__(self, features_in: int, inner_size: int, ca_num_heads: int, ca_num_layers: int,
                 self_attn_num_heads: int, self_attn_num_layers: int, depth: int):
        # N - MelSpec feature number D - small size
        super().__init__()
        self.latent_array = torch.rand(size=(features_in, inner_size), requires_grad=True)
        self.ca = TransformerDecoder(TransformerDecoderLayer(features_in, ca_num_heads, batch_first=True), 1)

        encoder_layer = TransformerEncoder(TransformerEncoderLayer(features_in, self_attn_num_heads, batch_first=True),
                                           self_attn_num_layers)
        self.transformer_encoders = torch.nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(depth)])
        decoder_layer = TransformerDecoder(TransformerDecoderLayer(features_in, ca_num_heads, batch_first=True),
                                           ca_num_layers)
        self.cross_attention_encoders = torch.nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(depth)])

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return OrderedDict(
            {
                "audio_signal":
                    NeuralType(('B', 'D', 'T'), SpectrogramType()),
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

    @typecheck()
    def forward(self, audio_signal, length):
        audio_signal = audio_signal.unsqueeze(1)
        batch_size, *axis, _ = audio_signal.shape
        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=audio_signal.device), axis))
        pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
        fourier_encoded = fourier_encode(pos, 16000)
        fourier_encoded = fourier_encoded.view(list(fourier_encoded.shape[:-2]) + [-1])
        fourier_encoded = torch.cat(batch_size * [fourier_encoded.unsqueeze(0)])
        audio_and_fe = torch.cat([fourier_encoded, audio_signal], dim=-1)
        byte_array = torch.cat(batch_size * [self.latent_array.unsqueeze(0).to(audio_signal.device)]).swapaxes(-2, -1)
        audio_and_fe = audio_and_fe.permute(0, -1, 1, 2).flatten(2)
        byte_array_mask = torch.zeros(size=(byte_array.shape[1], byte_array.shape[1]),
                                      device=audio_signal.device).bool()
        audio_signal_mask = self.make_pad_mask(length, max_time=audio_and_fe.shape[1], device=audio_signal.device)
        crossed = self.ca(byte_array.to(audio_signal.device), audio_and_fe.to(audio_signal.device),
                          tgt_mask=byte_array_mask, memory_key_padding_mask=audio_signal_mask)

        for self_att, cross_att in zip(self.transformer_encoders, self.cross_attention_encoders):
            residual = crossed
            crossed = cross_att(
                byte_array, audio_and_fe, tgt_mask=byte_array_mask, memory_key_padding_mask=audio_signal_mask
            )
            crossed = self_att(crossed, byte_array_mask)
            crossed += residual
        return crossed, torch.zeros(size=(batch_size,)).int() + self.latent_array.shape[0]

    @staticmethod
    def make_pad_mask(seq_lens, max_time, device=None):
        """Make masking for padding."""
        bs = seq_lens.size(0)
        seq_range = torch.arange(0, max_time, dtype=torch.int32)
        seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_time)
        seq_lens = seq_lens.type(seq_range_expand.dtype).to(seq_range_expand.device)
        seq_length_expand = seq_lens.unsqueeze(-1)
        mask = seq_range_expand < seq_length_expand

        if device:
            mask = mask.to(device)
        return mask
