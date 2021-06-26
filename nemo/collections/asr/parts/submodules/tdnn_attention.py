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
#
import torch
from numpy import inf
from torch import nn as nn
from torch.nn import functional as F

from nemo.collections.asr.parts.submodules.jasper import get_same_padding, init_weights


class StatsPoolLayer(nn.Module):
    def __init__(self, feat_in, pool_mode='xvector'):
        super().__init__()
        self.pool_mode = pool_mode
        self.feat_in = feat_in
        if self.pool_mode == 'xvector':
            self.feat_in += feat_in
        elif self.pool_mode == 'tap':
            self.feat_in = feat_in
        else:
            raise ValueError("pool mode for stats must be either tap or xvector based")

    def forward(self, encoder_output, length=None):
        mean = encoder_output.mean(dim=-1)  # Time Axis
        if self.pool_mode == 'xvector':
            std = encoder_output.std(dim=-1)
            pooled = torch.cat([mean, std], dim=-1)
        else:
            pooled = mean
        return pooled


def lens_to_mask(lens, max_len, device=None):
    lens_mat = torch.arange(max_len).to(device)
    mask = lens_mat[:max_len].unsqueeze(0) < lens.unsqueeze(1)
    mask = mask.unsqueeze(1)
    num_values = torch.sum(mask, dim=2, keepdim=True)
    return mask, num_values


def get_statistics_with_mask(x, m, dim=2, eps=1e-10):
    mean = torch.sum((m * x), dim=dim)
    std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
    return mean, std


class TDNN_Module(nn.Module):
    def __init__(self, inp_filters, out_filters, kernel_size=1, dilation=1, stride=1, padding=None):
        super(TDNN_Module, self).__init__()
        if padding is None:
            padding = get_same_padding(kernel_size, stride=stride, dilation=dilation)

        self.conv_layer = nn.Conv1d(
            in_channels=inp_filters,
            out_channels=out_filters,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_filters)

    def forward(self, x, length=None):
        x = self.conv_layer(x)
        x = self.activation(x)
        return self.bn(x)


class SE_Module(nn.Module):
    def __init__(self, inp_filters, se_filters, out_filters, kernel_size=1, dilation=1):
        super(SE_Module, self).__init__()
        self.se_layer = nn.Sequential(
            nn.Conv1d(inp_filters, se_filters, kernel_size=kernel_size, dilation=dilation,),
            nn.ReLU(),
            nn.BatchNorm1d(se_filters),
            nn.Conv1d(se_filters, out_filters, kernel_size=kernel_size, dilation=dilation,),
            nn.Sigmoid(),
        )

    def forward(self, input, length=None):
        if length is None:
            x = torch.mean(input, dim=2, keep_dim=True)
        else:
            max_len = input.size(2)
            mask, num_values = lens_to_mask(length, max_len=max_len, device=input.device)
            x = torch.sum((input * mask), dim=2, keepdim=True) / (num_values)

        out = self.se_layer(x)
        return out * input


class SE_TDNN_Module(nn.Module):
    """
    Modified SE_TDNN group module from ECAPA implementation for faster training and inference
    """

    def __init__(
        self,
        inp_filters,
        out_filters,
        group_scale=8,
        se_channels=128,
        kernel_size=1,
        dilation=1,
        group_kernel_size=3,
        init_mode='xavier_uniform',
    ):
        super(SE_TDNN_Module, self).__init__()
        self.out_filters = out_filters
        padding_val = get_same_padding(kernel_size=group_kernel_size, dilation=dilation, stride=1)

        group_conv = nn.Conv1d(
            out_filters,
            out_filters,
            kernel_size=group_kernel_size,
            dilation=dilation,
            padding=padding_val,
            groups=group_scale,
        )
        self.conv_block = nn.Sequential(
            TDNN_Module(inp_filters, out_filters, kernel_size=kernel_size, dilation=dilation),
            group_conv,
            nn.ReLU(),
            nn.BatchNorm1d(out_filters),
            TDNN_Module(out_filters, out_filters, kernel_size=kernel_size, dilation=dilation),
        )

        self.se_layer = SE_Module(out_filters, se_channels, out_filters)

        self.apply(lambda x: init_weights(x, mode=init_mode))

    def forward(self, input, length=None):
        x = self.conv_block(input)
        x = self.se_layer(x, length)
        return x + input


class AttentivePoolLayer(nn.Module):
    def __init__(self, inp_filters, attention_channels=128, kernel_size=1, dilation=1, eps=1e-10):
        super(AttentivePoolLayer, self).__init__()

        self.feat_in = 2 * inp_filters

        self.attention_layer = nn.Sequential(
            TDNN_Module(inp_filters * 3, attention_channels, kernel_size=kernel_size, dilation=dilation),
            nn.Tanh(),
            nn.Conv1d(
                in_channels=attention_channels, out_channels=inp_filters, kernel_size=kernel_size, dilation=dilation,
            ),
        )
        self.norm_bn = nn.BatchNorm1d(self.feat_in)
        self.eps = eps

    def forward(self, x, length=None):
        max_len = x.size(2)

        if length is None:
            length = torch.ones(x.shape[0], device=x.device)

        mask, num_values = lens_to_mask(length, max_len=max_len, device=x.device)

        # encoder statistics
        mean, std = get_statistics_with_mask(x, mask / num_values)
        mean = mean.unsqueeze(2).repeat(1, 1, max_len)
        std = std.unsqueeze(2).repeat(1, 1, max_len)
        attn = torch.cat([x, mean, std], dim=1)

        # attention statistics
        attn = self.attention_layer(attn)  # attention pass
        attn = attn.masked_fill(mask == 0, -inf)
        alpha = F.softmax(attn, dim=2)  # attention values, α
        mu, sg = get_statistics_with_mask(x, alpha)  # µ and ∑

        # gather
        return torch.cat((mu, sg), dim=1).unsqueeze(2)
