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


class StackingSubsampling(torch.nn.Module):
    def __init__(self, subsampling_factor, feat_in, feat_out):
        super(StackingSubsampling, self).__init__()
        self.subsampling_factor = subsampling_factor
        self.proj_out = torch.nn.Linear(subsampling_factor * feat_in, feat_out)

    def forward(self, x, lengths, cache=None, cache_next=None):
        b, t, h = x.size()
        pad_size = self.subsampling_factor - (t % self.subsampling_factor)
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
        _, t, _ = x.size()
        x = torch.reshape(x, (b, t // self.subsampling_factor, h * self.subsampling_factor))
        x = self.proj_out(x)
        lengths = torch.div(lengths + pad_size, self.subsampling_factor, rounding_mode='floor')
        return x, lengths


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
        if padding == -1:
            self._left_padding = kernel_size - 1
            self._right_padding = stride - 1
        else:
            self._left_padding = padding
            self._right_padding = padding

        self._stride = stride
        self._max_cache_len = kernel_size - stride
        # self._max_cache_len = 2 * stride + 1
        self._ignore_len = self._right_padding // self._stride
        # self._ignore_len = 0

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

    def forward(self, x, cache=None, cache_next=None):
        if type(x) == tuple:
            if cache is not None:
                raise ValueError("Cache can not be non-None when input x is a tuple!")
            x, cache = x

        input_x = x
        x_length = x.size()[2]
        if cache is None:
            x = F.pad(x, pad=(self._left_padding, self._right_padding, self._left_padding, self._right_padding))
        else:
            if hasattr(self, '_cache_id'):
                cache = cache[self._cache_id]
                cache_next = cache_next[self._cache_id]

            cache_length = cache.size()[-2]
            cache_next_length = cache.size()[-2]
            needed_cache = cache[:, :, -self._max_cache_len :, -self._max_cache_len :]
            x = torch.cat((needed_cache, x), dim=-1)

        x = super().forward(x)
        if cache is None:
            if self._ignore_len > 0:
                # x = x[:, :, :-(self.padding[0]//2), :-(self.padding[0]//2)]
                x = x[:, :, : -self._ignore_len, : -self._ignore_len]
        else:
            cache_next[:, :, :-x_length] = cache[:, :, -(cache_next_length - x_length) :]
            cache_next[:, :, -x_length:] = input_x

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
        self.is_streaming = False

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
                self._left_padding = self._kernel_size - 1
                self._right_padding = self._stride - 1
                self._max_cache_len = 5  # calculate it automatically
            else:
                self._left_padding = (self._kernel_size - 1) // 2
                self._right_padding = (self._kernel_size - 1) // 2
                self._max_cache_len = 0

            for i in range(self._sampling_num):
                if self.is_causal:
                    layers.append(
                        CausalConv2D(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            # padding=self._kernel_size - 1,
                            padding=-1,
                        )
                    )
                else:
                    layers.append(
                        torch.nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=conv_channels,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._left_padding,
                        )
                    )
                layers.append(activation)
                in_channels = conv_channels
        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        in_length = torch.tensor(feat_in, dtype=torch.float)
        out_length = calc_length(
            lengths=in_length,
            padding=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x, lengths): #, cache=None, cache_next=None):
        lengths = calc_length(
            lengths,
            padding=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )

        x = x.unsqueeze(1)
        #x_length = x.size()[-2]
        # if self.is_streaming: #cache is not None:
        #     # input_x = x
        #     # if hasattr(self, '_cache_id'):
        #     #     cache = cache[self._cache_id]
        #     #     cache_next = cache_next[self._cache_id]
        #     # if x_length != 1:
        #     #     # needed_cache = cache[:, :, -self._max_cache_len :, -self._max_cache_len :]
        #     #     needed_cache = cache[:, :, -self._max_cache_len:]
        #     #     x = torch.cat((needed_cache, x), dim=-2)
        #     #
        #     # cache_length = cache.size()[-2]
        #     # cache_next_length = cache_next.size()[-2]
        #     # cache_next[:, :, :-x_length] = cache[:, :, cache_length - (cache_next_length - x_length) :]
        #     # cache_next[:, :, -x_length:, :] = input_x[:, :, -cache_next_length:, :]
        #     conv_right_padding = 0
        # else:
        #conv_right_padding = self._right_padding

        x = self.conv(x)
        #if cache is not None and x_length != 1:

        if self.is_streaming: # and x.size(2) > 1: #cache is not None:
            x = x[:, :, 2:, :]
            lengths = lengths - 2

            # cache_next[:, :, :-x_length] = cache[:, :, -(cache_length - x_length) :].clone()
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))

        return x, lengths


def calc_length(lengths, padding, kernel_size, stride, ceil_mode, repeat_num=1):
    """ Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = padding - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)
