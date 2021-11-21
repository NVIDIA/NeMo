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
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv2D(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ) -> None:
        self._padding = padding
        self._stride = stride
        self._needed_cache_len = kernel_size - stride
        self._ignore_len = self._padding // self._stride

        padding = 0
        super(CausalConv2D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    def forward(self, x):
        if type(x) == tuple:
            x, cache = x
        else:
            cache = None
        input_x = x
        x_length = x.size()[2]
        if cache is None:
            x = F.pad(x, pad=(self._padding, self._padding, self._padding, self._padding))
        else:
            if not hasattr(self, 'cache_id'):
                cache = cache[self._cache_id]

            cache_length = cache.size()[-1]
            needed_cache = cache[:, :, -self._needed_cache_len :, -self._needed_cache_len :]
            x = torch.cat((needed_cache, x), dim=-1)

        x = super().forward(x)
        if cache is None:
            # x = x[:, :, :-(self.padding[0]//2), :-(self.padding[0]//2)]
            x = x[:, :, : -self._ignore_len, : -self._ignore_len]
        else:
            cache[:, :, :-x_length] = cache[:, :, -(cache_length - x_length) :]
            cache[:, :, -x_length:] = input_x

        return x


class ConvSubsampling(torch.nn.Module):
    """Convolutional subsampling which supports VGGNet and striding approach introduced in:
    VGGNet Subsampling: https://arxiv.org/pdf/1910.12977.pdf
    Striding Subsampling:
        "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition" by Linhao Dong et al.
    Args:
        subsampling (str): The subsampling technique from {"vggnet", "striding"}
        subsampling_factor (int): The subsampling factor which should be a power of 2
        feat_in (int): size of the input features
        feat_out (int): size of the output features
        conv_channels (int): Number of channels for the convolution layers.
        activation (Module): activation function, default is nn.ReLU()
    """

    def __init__(
        self, subsampling, subsampling_factor, feat_in, feat_out, conv_channels, activation=nn.ReLU(), is_causal=False
    ):
        super(ConvSubsampling, self).__init__()
        self._subsampling = subsampling

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))

        # is_causal = False
        self.is_causal = is_causal

        in_channels = 1
        layers = []
        if subsampling == 'vggnet':
            self._padding = 0
            self._stride = 2
            self._kernel_size = 2
            self._ceil_mode = True

            for i in range(self._sampling_num):
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1
                    )
                )
                layers.append(activation)
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1
                    )
                )
                layers.append(activation)
                layers.append(
                    torch.nn.MaxPool2d(
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._padding,
                        ceil_mode=self._ceil_mode,
                    )
                )
                in_channels = conv_channels
        elif subsampling == 'striding':
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            if self.is_causal:
                self._padding = self._kernel_size - 1
            else:
                self._padding = (self._kernel_size - 1) // 2

            for i in range(self._sampling_num):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._kernel_size - 1,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._padding,
                        )
                    )
                layers.append(activation)
                in_channels = conv_channels
        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        in_length = torch.tensor(feat_in, dtype=torch.int)
        for i in range(self._sampling_num):
            # length=int(in_length),
            out_length = calc_length(
                lengths=in_length,
                padding=self._padding,
                kernel_size=self._kernel_size,
                stride=self._stride,
                ceil_mode=self._ceil_mode,
            )
            in_length = out_length
            if self.is_causal:
                out_length -= self._padding // self._stride

        out_length = int(out_length)
        self.out = torch.nn.Linear(conv_channels * out_length, feat_out)
        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x, lengths, cache=None):
        x = x.unsqueeze(1)
        x = self.conv((x, cache))
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))

        # TODO: improve the performance of length calculation
        new_lengths = lengths
        for i in range(self._sampling_num):
            new_lengths = calc_length(
                lengths=new_lengths,
                padding=self._padding,
                kernel_size=self._kernel_size,
                stride=self._stride,
                ceil_mode=self._ceil_mode,
            )
            if self.is_causal:
                new_lengths -= self._padding // self._stride
        # print(new_lengths)
        # print(x.size()[1])
        # assert new_lengths == x.size()[1]
        new_lengths = new_lengths.to(dtype=lengths.dtype)
        return x, new_lengths


def calc_length(lengths, padding, kernel_size, stride, ceil_mode):
    """ Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    if ceil_mode:
        lengths = torch.ceil((lengths + (2 * padding) - (kernel_size - 1) - 1) / float(stride) + 1)
    else:
        lengths = torch.floor(torch.div(lengths + (2 * padding) - (kernel_size - 1) - 1, stride) + 1)
    return lengths
