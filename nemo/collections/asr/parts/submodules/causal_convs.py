# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Union

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['CausalConv2D', 'CausalConv1D']


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
        if padding is None:
            self._left_padding = kernel_size - 1
            self._right_padding = stride - 1
        else:
            self._left_padding = padding
            self._right_padding = padding

        self._stride = stride
        self._max_cache_len = kernel_size - stride
        self._ignore_len = self._right_padding // self._stride

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

    def forward(
        self, x,
    ):
        x = F.pad(x, pad=(self._left_padding, self._right_padding, self._left_padding, self._right_padding))
        x = super().forward(x)
        return x


class CausalConv1D(nn.Conv1d):
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
        self.cache_drop_size = None
        if padding is None:
            self._left_padding = kernel_size - 1
            self._right_padding = stride - 1
        else:
            if stride != 1 and padding != kernel_size - 1:
                raise ValueError("No striding allowed for non-symmetric convolutions!")
            if isinstance(padding, int):
                self._left_padding = padding
                self._right_padding = padding
            elif isinstance(padding, list) and len(padding) == 2:
                self._left_padding = padding[0]
                self._right_padding = padding[1]
            else:
                raise ValueError(f"Invalid padding param: {padding}!")

        self._max_cache_len = self._left_padding

        super(CausalConv1D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def do_caching(self, x, cache=None, cache_next=None):
        if cache is None:
            x = F.pad(x, pad=(self._left_padding, self._right_padding))
        else:
            input_x = x
            x_length = x.size(-1)
            if hasattr(self, '_cache_id'):
                cache = cache[self._cache_id]
                cache_next = cache_next[self._cache_id]
            cache_next_length = cache_next.size(-1)
            needed_cache = cache[:, :, -self._max_cache_len :]
            x = F.pad(x, pad=(0, self._right_padding))
            x = torch.cat((needed_cache, x), dim=-1)
        if cache_next is not None:
            x_keep_size = x_length - self.cache_drop_size
            cache_keep_size = min(x_keep_size, cache_next_length)
            cache_next[:, :, :-x_keep_size] = cache[:, :, cache_keep_size:]
            input_x_kept = input_x[:, :, :x_keep_size]
            cache_next[:, :, -cache_keep_size:] = input_x_kept[:, :, -cache_keep_size:]
        return x, cache_next

    def forward(self, x, cache=None, cache_next=None):
        x, cache_next = self.do_caching(x, cache=cache, cache_next=cache_next)
        x = super().forward(x)
        return x
