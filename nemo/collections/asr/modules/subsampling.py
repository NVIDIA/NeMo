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

import torch
import torch.nn as nn


class ConvSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    :param int idim: input dim
    :param int odim: output dim
    :param str activation: activation functions
    :param flaot dropout_rate: dropout rate

    """

    def __init__(self, subsampling, subsampling_factor, idim, odim, conv_channels=-1, activation=nn.ReLU()):
        super(ConvSubsampling, self).__init__()
        if conv_channels <= 0:
            conv_channels = odim
        self._subsampling = subsampling

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))

        in_channels = 1
        layers = []
        if subsampling == 'vggnet':
            self._padding = 0
            self._stride = 2
            self._kernel_size = 2
            self._ceil_mode = True  # TODO: is False better?

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
            self._padding = 1  # TODO: is 0 better?
            self._stride = 2
            self._kernel_size = 3
            self._ceil_mode = False

            for i in range(self._sampling_num):
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

        in_length = idim
        for i in range(self._sampling_num):
            out_length = calc_length(
                length=int(in_length),
                padding=self._padding,
                kernel_size=self._kernel_size,
                stride=self._stride,
                ceil_mode=self._ceil_mode,
            )
            in_length = out_length

        self.out = torch.nn.Linear(conv_channels * out_length, odim)
        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x, lengths):
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor or Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))

        # TODO: improve the performance of here
        new_lengths = lengths
        for i in range(self._sampling_num):
            new_lengths = [
                calc_length(
                    length=int(length),
                    padding=self._padding,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    ceil_mode=self._ceil_mode,
                )
                for length in new_lengths
            ]

        new_lengths = torch.IntTensor(new_lengths).to(lengths.device)
        return x, new_lengths


def calc_length(length, padding, kernel_size, stride, ceil_mode):
    if ceil_mode:
        length = math.ceil((length + (2 * padding) - (kernel_size - 1) - 1) / float(stride) + 1)
    else:
        length = math.floor((length + (2 * padding) - (kernel_size - 1) - 1) / float(stride) + 1)
    return length
