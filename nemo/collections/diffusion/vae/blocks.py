# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from torch import nn

try:
    from apex.contrib.group_norm import GroupNorm

    OPT_GROUP_NORM = True
except Exception:
    print('Fused optimized group norm has not been installed.')
    OPT_GROUP_NORM = False


def Normalize(in_channels, num_groups=32, act=""):
    """Creates a group normalization layer with specified activation.

    Args:
        in_channels (int): Number of channels in the input.
        num_groups (int, optional): Number of groups for GroupNorm. Defaults to 32.
        act (str, optional): Activation function name. Defaults to "".

    Returns:
        GroupNorm: A normalization layer with optional activation.
    """
    return GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True, act=act)


def nonlinearity(x):
    """Nonlinearity function used in temporal embedding projection.

    Currently implemented as a SiLU (Swish) function.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Output after applying SiLU activation.
    """
    return x * torch.sigmoid(x)


class ResnetBlock(nn.Module):
    """A ResNet-style block that can optionally apply a temporal embedding and shortcut projections.

    This block consists of two convolutional layers, normalization, and optional temporal embedding.
    It can adjust channel dimensions between input and output via shortcuts.
    """

    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, temb_channels=0):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int, optional): Number of output channels. Defaults to in_channels.
            conv_shortcut (bool, optional): Whether to use a convolutional shortcut. Defaults to False.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            temb_channels (int, optional): Number of channels in temporal embedding. Defaults to 0.
        """
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, act="silu")
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels, act="silu")
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        """Forward pass of the ResnetBlock.

        Args:
            x (Tensor): Input feature map of shape (B, C, H, W).
            temb (Tensor): Temporal embedding tensor of shape (B, temb_channels).

        Returns:
            Tensor: Output feature map of shape (B, out_channels, H, W).
        """
        h = x
