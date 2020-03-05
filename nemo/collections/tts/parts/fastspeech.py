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

import collections
import os

import numpy as np
import torch
from torch import nn


class FastSpeechDataset:
    def __init__(self, audio_dataset, durs_dir):
        self._audio_dataset = audio_dataset
        self._durs_dir = durs_dir

    def __getitem__(self, index):
        audio, audio_len, text, text_len = self._audio_dataset[index]
        dur_true = torch.tensor(np.load(os.path.join(self._durs_dir, f'{index}.npy'))).long()
        return dict(audio=audio, audio_len=audio_len, text=text, text_len=text_len, dur_true=dur_true)

    def __len__(self):
        return len(self._audio_dataset)


class LengthRegulator(nn.Module):
    """Length Regulator."""

    def __init__(self, encoder_output_size, duration_predictor_filter_size, duration_predictor_kernel_size, dropout):
        super(LengthRegulator, self).__init__()

        self.duration_predictor = DurationPredictor(
            input_size=encoder_output_size,
            filter_size=duration_predictor_filter_size,
            kernel=duration_predictor_kernel_size,
            conv_output_size=duration_predictor_filter_size,
            dropout=dropout,
        )

    def forward(self, encoder_output, encoder_output_mask, target=None, alpha=1.0, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(encoder_output, encoder_output_mask)

        if self.training:
            output, dec_pos = self.get_output(encoder_output, target, alpha, mel_max_length)
        else:
            duration_predictor_output = torch.clamp_min(torch.exp(duration_predictor_output) - 1, 0)

            output, dec_pos = self.get_output(encoder_output, duration_predictor_output, alpha)

        return output, dec_pos, duration_predictor_output

    @staticmethod
    def get_output(encoder_output, duration_predictor_output, alpha, mel_max_length=None):
        output = list()
        dec_pos = list()

        for i in range(encoder_output.size(0)):
            repeats = duration_predictor_output[i].float() * alpha
            repeats = torch.round(repeats).long()
            output.append(torch.repeat_interleave(encoder_output[i], repeats, dim=0))
            dec_pos.append(torch.from_numpy(np.indices((output[i].shape[0],))[0] + 1))

        output = torch.nn.utils.rnn.pad_sequence(output, batch_first=True)
        dec_pos = torch.nn.utils.rnn.pad_sequence(dec_pos, batch_first=True)

        dec_pos = dec_pos.to(output.device, non_blocking=True)

        if mel_max_length:
            output = output[:, :mel_max_length]
            dec_pos = dec_pos[:, :mel_max_length]

        return output, dec_pos


class ConvTranspose(nn.Module):
    """Convolution Module with transposes of last two dimensions."""

    def __init__(
        self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, w_init='relu'
    ):
        super(ConvTranspose, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


class DurationPredictor(nn.Module):
    """Duration Predictor."""

    def __init__(self, input_size, filter_size, kernel, conv_output_size, dropout):
        super(DurationPredictor, self).__init__()

        self.input_size = input_size
        self.filter_size = filter_size
        self.kernel = kernel
        self.conv_output_size = conv_output_size
        self.dropout = dropout

        self.conv_layer = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "conv1d_1",
                        ConvTranspose(self.input_size, self.filter_size, kernel_size=self.kernel, padding=1),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        ConvTranspose(self.filter_size, self.filter_size, kernel_size=self.kernel, padding=1),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1, bias=True)

    def forward(self, encoder_output, encoder_output_mask):
        encoder_output = encoder_output * encoder_output_mask

        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out * encoder_output_mask
        out = out.squeeze(-1)

        return out
