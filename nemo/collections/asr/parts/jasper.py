# Copyright (C) NVIDIA CORPORATION. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

jasper_activations = {
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
}


def init_weights(m, mode='xavier_uniform'):
    if isinstance(m, MaskedConv1d):
        init_weights(m.conv, mode)
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        if mode == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif mode == 'xavier_normal':
            nn.init.xavier_normal_(m.weight, gain=1.0)
        elif mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif mode == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        else:
            raise ValueError("Unknown Initialization mode: {0}".format(mode))
    elif isinstance(m, nn.BatchNorm1d):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def compute_new_kernel_size(kernel_size, kernel_width):
    new_kernel_size = max(int(kernel_size * kernel_width), 1)
    # If kernel is even shape, round up to make it odd
    if new_kernel_size % 2 == 0:
        new_kernel_size += 1
    return new_kernel_size


def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    if dilation > 1:
        return (dilation * kernel_size) // 2 - 1
    return kernel_size // 2


class StatsPoolLayer(nn.Module):
    def __init__(self, gram=False, super_vector=False):
        super().__init__()
        self.gram = gram
        self.super = super_vector

    def forward(self, encoder_output):

        mean = encoder_output.mean(dim=-1)  # Time Axis
        std = encoder_output.std(dim=-1)

        pooled = torch.cat([mean, std], dim=-1)

        if self.gram:
            time_len = encoder_output.shape[-1]
            # encoder_output = encoder_output
            cov = encoder_output.bmm(encoder_output.transpose(2, 1))  # cov matrix
            cov = cov.view(cov.shape[0], -1) / (time_len)

        if self.gram and not self.super:
            return cov

        if self.super and self.gram:
            pooled = torch.cat([pooled, cov], dim=-1)

        return pooled


class MaskedConv1d(nn.Module):
    __constants__ = ["use_conv_mask", "real_out_channels", "heads"]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        heads=-1,
        bias=False,
        use_mask=True,
    ):
        super(MaskedConv1d, self).__init__()

        if not (heads == -1 or groups == in_channels):
            raise ValueError("Only use heads for depthwise convolutions")

        self.real_out_channels = out_channels
        if heads != -1:
            in_channels = heads
            out_channels = heads
            groups = heads

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.use_mask = use_mask
        self.heads = heads

    def get_seq_len(self, lens):
        return (
            lens + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1
        ) / self.conv.stride[0] + 1

    def forward(self, x, lens):
        if self.use_mask:
            lens = lens.to(dtype=torch.long)
            max_len = x.size(2)
            mask = torch.arange(max_len).to(lens.device).expand(len(lens), max_len) >= lens.unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
            # del mask
            lens = self.get_seq_len(lens)

        sh = x.shape
        if self.heads != -1:
            x = x.view(-1, self.heads, sh[-1])

        out = self.conv(x)

        if self.heads != -1:
            out = out.view(sh[0], self.real_out_channels, -1)

        return out, lens


class GroupShuffle(nn.Module):
    def __init__(self, groups, channels):
        super(GroupShuffle, self).__init__()

        self.groups = groups
        self.channels_per_group = channels // groups

    def forward(self, x):
        sh = x.shape

        x = x.view(-1, self.groups, self.channels_per_group, sh[-1])

        x = torch.transpose(x, 1, 2).contiguous()

        x = x.view(-1, self.groups * self.channels_per_group, sh[-1])

        return x


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int,
        context_window: int = -1,
        interpolation_mode: str = 'nearest',
        activation: Optional[Callable] = None,
    ):
        """
        Squeeze-and-Excitation sub-module.

        Args:
            channels: Input number of channels.
            reduction_ratio: Reduction ratio for "squeeze" layer.
            context_window: Integer number of timesteps that the context
                should be computed over, using stride 1 average pooling.
                If value < 1, then global context is computed.
            interpolation_mode: Interpolation mode of timestep dimension.
                Used only if context window is > 1.
                The modes available for resizing are: `nearest`, `linear` (3D-only),
                `bilinear`, `area`
            activation: Intermediate activation function used. Must be a
                callable activation function.
        """
        super(SqueezeExcite, self).__init__()
        self.context_window = int(context_window)
        self.interpolation_mode = interpolation_mode

        if self.context_window <= 0:
            self.pool = nn.AdaptiveAvgPool1d(1)  # context window = T
        else:
            self.pool = nn.AvgPool1d(self.context_window, stride=1)

        if activation is None:
            activation = nn.ReLU(inplace=True)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            activation,
            nn.Linear(channels // reduction_ratio, channels, bias=False),
        )

    def forward(self, x):
        batch, channels, timesteps = x.size()
        y = self.pool(x)  # [B, C, T - context_window + 1]
        y = y.transpose(1, 2)  # [B, T - context_window + 1, C]
        y = self.fc(y)  # [B, T - context_window + 1, C]
        y = y.transpose(1, 2)  # [B, C, T - context_window + 1]

        if self.context_window > 0:
            y = torch.nn.functional.interpolate(y, size=timesteps, mode=self.interpolation_mode)

        y = torch.sigmoid(y)

        return x * y


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class JasperBlock(nn.Module):
    __constants__ = ["conv_mask", "separable", "residual_mode", "res", "mconv"]

    def __init__(
        self,
        inplanes,
        planes,
        repeat=3,
        kernel_size=11,
        kernel_size_factor=1,
        stride=1,
        dilation=1,
        padding='same',
        dropout=0.2,
        activation=None,
        residual=True,
        groups=1,
        separable=False,
        heads=-1,
        normalization="batch",
        norm_groups=1,
        residual_mode='add',
        residual_panes=[],
        conv_mask=False,
        se=False,
        se_reduction_ratio=16,
        se_context_window=None,
        se_interpolation_mode='nearest',
        stride_last=False,
    ):
        super(JasperBlock, self).__init__()

        if padding != "same":
            raise ValueError("currently only 'same' padding is supported")

        kernel_size_factor = float(kernel_size_factor)
        if type(kernel_size) in (list, tuple):
            kernel_size = [compute_new_kernel_size(k, kernel_size_factor) for k in kernel_size]
        else:
            kernel_size = compute_new_kernel_size(kernel_size, kernel_size_factor)

        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])
        self.conv_mask = conv_mask
        self.separable = separable
        self.residual_mode = residual_mode
        self.se = se

        inplanes_loop = inplanes
        conv = nn.ModuleList()

        for _ in range(repeat - 1):
            # Stride last means only the last convolution in block will have stride
            if stride_last:
                stride_val = [1]
            else:
                stride_val = stride

            conv.extend(
                self._get_conv_bn_layer(
                    inplanes_loop,
                    planes,
                    kernel_size=kernel_size,
                    stride=stride_val,
                    dilation=dilation,
                    padding=padding_val,
                    groups=groups,
                    heads=heads,
                    separable=separable,
                    normalization=normalization,
                    norm_groups=norm_groups,
                )
            )

            conv.extend(self._get_act_dropout_layer(drop_prob=dropout, activation=activation))

            inplanes_loop = planes

        conv.extend(
            self._get_conv_bn_layer(
                inplanes_loop,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding_val,
                groups=groups,
                heads=heads,
                separable=separable,
                normalization=normalization,
                norm_groups=norm_groups,
            )
        )

        if se:
            conv.append(
                SqueezeExcite(
                    planes,
                    reduction_ratio=se_reduction_ratio,
                    context_window=se_context_window,
                    interpolation_mode=se_interpolation_mode,
                    activation=activation,
                )
            )

        self.mconv = conv

        res_panes = residual_panes.copy()
        self.dense_residual = residual

        if residual:
            res_list = nn.ModuleList()

            if residual_mode == 'stride_add':
                stride_val = stride
            else:
                stride_val = [1]

            if len(residual_panes) == 0:
                res_panes = [inplanes]
                self.dense_residual = False
            for ip in res_panes:
                res = nn.ModuleList(
                    self._get_conv_bn_layer(
                        ip,
                        planes,
                        kernel_size=1,
                        normalization=normalization,
                        norm_groups=norm_groups,
                        stride=stride_val,
                    )
                )

                res_list.append(res)

            self.res = res_list
        else:
            self.res = None

        self.mout = nn.Sequential(*self._get_act_dropout_layer(drop_prob=dropout, activation=activation))

    def _get_conv(
        self,
        in_channels,
        out_channels,
        kernel_size=11,
        stride=1,
        dilation=1,
        padding=0,
        bias=False,
        groups=1,
        heads=-1,
        separable=False,
    ):
        use_mask = self.conv_mask
        if use_mask:
            return MaskedConv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=bias,
                groups=groups,
                heads=heads,
                use_mask=use_mask,
            )
        else:
            return nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=bias,
                groups=groups,
            )

    def _get_conv_bn_layer(
        self,
        in_channels,
        out_channels,
        kernel_size=11,
        stride=1,
        dilation=1,
        padding=0,
        bias=False,
        groups=1,
        heads=-1,
        separable=False,
        normalization="batch",
        norm_groups=1,
    ):
        if norm_groups == -1:
            norm_groups = out_channels

        if separable:
            layers = [
                self._get_conv(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                    groups=in_channels,
                    heads=heads,
                ),
                self._get_conv(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    dilation=1,
                    padding=0,
                    bias=bias,
                    groups=groups,
                ),
            ]
        else:
            layers = [
                self._get_conv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                    groups=groups,
                )
            ]

        if normalization == "group":
            layers.append(nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels))
        elif normalization == "instance":
            layers.append(nn.GroupNorm(num_groups=out_channels, num_channels=out_channels))
        elif normalization == "layer":
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        elif normalization == "batch":
            layers.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1))
        else:
            raise ValueError(
                f"Normalization method ({normalization}) does not match" f" one of [batch, layer, group, instance]."
            )

        if groups > 1:
            layers.append(GroupShuffle(groups, out_channels))
        return layers

    def _get_act_dropout_layer(self, drop_prob=0.2, activation=None):
        if activation is None:
            activation = nn.Hardtanh(min_val=0.0, max_val=20.0)
        layers = [activation, nn.Dropout(p=drop_prob)]
        return layers

    def forward(self, input_: Tuple[List[Tensor], Optional[Tensor]]):
        # type: (Tuple[List[Tensor], Optional[Tensor]]) -> Tuple[List[Tensor], Optional[Tensor]] # nopep8
        lens_orig = None
        xs = input_[0]
        if len(input_) == 2:
            xs, lens_orig = input_

        # compute forward convolutions
        out = xs[-1]

        lens = lens_orig
        for i, l in enumerate(self.mconv):
            # if we're doing masked convolutions, we need to pass in and
            # possibly update the sequence lengths
            # if (i % 4) == 0 and self.conv_mask:
            if isinstance(l, MaskedConv1d):
                out, lens = l(out, lens)
            else:
                out = l(out)

        # compute the residuals
        if self.res is not None:
            for i, layer in enumerate(self.res):
                res_out = xs[i]
                for j, res_layer in enumerate(layer):
                    if isinstance(res_layer, MaskedConv1d):
                        res_out, _ = res_layer(res_out, lens_orig)
                    else:
                        res_out = res_layer(res_out)

                if self.residual_mode == 'add' or self.residual_mode == 'stride_add':
                    out = out + res_out
                else:
                    out = torch.max(out, res_out)

        # compute the output
        out = self.mout(out)
        if self.res is not None and self.dense_residual:
            return xs + [out], lens

        return [out], lens


# Register swish activation function
jasper_activations['swish'] = Swish
