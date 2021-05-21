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
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import _calculate_correct_fan
from torch.nn.modules.utils import _single

from nemo.collections.asr.parts.activations import Swish
from nemo.utils import logging

try:
    from pytorch_quantization import calib
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import quant_modules
    from pytorch_quantization.tensor_quant import QuantDescriptor

    PYTORCH_QUANTIZATION_AVAILABLE = True
except ImportError:
    PYTORCH_QUANTIZATION_AVAILABLE = False

jasper_activations = {"hardtanh": nn.Hardtanh, "relu": nn.ReLU, "selu": nn.SELU, "swish": Swish, "silu": nn.SiLU}


def tds_uniform_(tensor, mode='fan_in'):
    """
    Uniform Initialization from the paper [Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2460.pdf)
    Normalized to -

    .. math::
        \text{bound} = \text{2} \times \sqrt{\frac{1}{\text{fan\_mode}}}

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = 2.0  # sqrt(4.0) = 2
    std = gain / math.sqrt(fan)  # sqrt(4.0 / fan_in)
    bound = std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def tds_normal_(tensor, mode='fan_in'):
    """
    Normal Initialization from the paper [Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2460.pdf)
    Normalized to -

    .. math::
        \text{bound} = \text{2} \times \sqrt{\frac{1}{\text{fan\_mode}}}

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = 2.0
    std = gain / math.sqrt(fan)  # sqrt(4.0 / fan_in)
    bound = std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.normal_(0.0, bound)


def init_weights(m, mode: Optional[str] = 'xavier_uniform'):
    if isinstance(m, MaskedConv1d):
        init_weights(m.conv, mode)
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        if mode is not None:
            if mode == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif mode == 'xavier_normal':
                nn.init.xavier_normal_(m.weight, gain=1.0)
            elif mode == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif mode == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif mode == 'tds_uniform':
                tds_uniform_(m.weight)
            elif mode == 'tds_normal':
                tds_normal_(m.weight)
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


def get_same_padding(kernel_size, stride, dilation) -> int:
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (dilation * (kernel_size - 1)) // 2


def get_asymtric_padding(kernel_size, stride, dilation, future_context):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")

    left_context = kernel_size - 1 - future_context
    right_context = future_context

    symmetric_padding = get_same_padding(kernel_size, stride, dilation)

    if kernel_size <= future_context:
        # kernel size is smaller than future context, equivalent to using entire context of kernel
        # simply return symmetric padding for this scenario
        logging.warning(
            f"Future context window is larger than the kernel size!\n"
            f"Left context = {left_context} | Right context = greater than {right_context} | "
            f"Kernel size = {kernel_size}\n"
            f"Switching to symmetric padding (left context = right context = {symmetric_padding})"
        )
        return symmetric_padding

    if left_context < symmetric_padding:
        logging.warning(
            f"Future context window is larger than half the kernel size!\n"
            f"Conv layer therefore uses more future information than past to compute its output!\n"
            f"Left context = {left_context} | Right context = {right_context} | "
            f"Kernel size = {kernel_size}"
        )

    if dilation > 1:
        left_context = dilation * kernel_size - 1 - dilation * future_context
        right_context = dilation * future_context
        return (left_context, right_context)

    return (left_context, right_context)


class StatsPoolLayer(nn.Module):
    def __init__(self, feat_in, pool_mode='xvector'):
        super().__init__()
        self.feat_in = 0
        if pool_mode == 'gram':
            gram = True
            super_vector = False
        elif pool_mode == 'superVector':
            gram = True
            super_vector = True
        else:
            gram = False
            super_vector = False

        if gram:
            self.feat_in += feat_in ** 2
        else:
            self.feat_in += 2 * feat_in

        if super_vector and gram:
            self.feat_in += 2 * feat_in

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
            cov = cov.view(cov.shape[0], -1) / time_len

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
        quantize=False,
    ):
        super(MaskedConv1d, self).__init__()

        if not (heads == -1 or groups == in_channels):
            raise ValueError("Only use heads for depthwise convolutions")

        self.real_out_channels = out_channels
        if heads != -1:
            in_channels = heads
            out_channels = heads
            groups = heads

        # preserve original padding
        self._padding = padding

        # if padding is a tuple/list, it is considered as asymmetric padding
        if type(padding) in (tuple, list):
            self.pad_layer = nn.ConstantPad1d(padding, value=0.0)
            # reset padding for conv since pad_layer will handle this
            padding = 0
        else:
            self.pad_layer = None

        if PYTORCH_QUANTIZATION_AVAILABLE and quantize:
            self.conv = quant_nn.QuantConv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        elif not PYTORCH_QUANTIZATION_AVAILABLE and quantize:
            raise ImportError(
                "pytorch-quantization is not installed. Install from "
                "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
            )
        else:
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

        # Calculations for "same" padding cache
        self.same_padding = (self.conv.stride[0] == 1) and (
            2 * self.conv.padding[0] == self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
        )
        if self.pad_layer is None:
            self.same_padding_asymmetric = False
        else:
            self.same_padding_asymmetric = (self.conv.stride[0] == 1) and (
                sum(self._padding) == self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
            )

        # `self.lens` caches consecutive integers from 0 to `self.max_len` that are used to compute the mask for a
        # batch. Recomputed to bigger size as needed. Stored on a device of the latest batch lens.
        if self.use_mask:
            self.max_len = 0
            self.lens = None

    def get_seq_len(self, lens):
        if self.same_padding or self.same_padding_asymmetric:
            return lens

        if self.pad_layer is None:
            return (
                lens + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1
            ) // self.conv.stride[0] + 1
        else:
            return (
                lens + sum(self._padding) - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1
            ) // self.conv.stride[0] + 1

    def forward(self, x, lens):
        if self.use_mask:
            max_len = x.size(2)
            if max_len > self.max_len:
                self.lens = torch.arange(max_len)
                self.max_len = max_len

            self.lens = self.lens.to(lens.device)
            mask = self.lens[:max_len].unsqueeze(0) < lens.unsqueeze(1)
            x = x * mask.unsqueeze(1).to(device=x.device)
            lens = self.get_seq_len(lens)

        # asymmtric pad if necessary
        if self.pad_layer is not None:
            x = self.pad_layer(x)

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
        quantize: bool = False,
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
        self.interpolation_mode = interpolation_mode
        self._quantize = quantize

        self.pool = None  # prepare a placeholder which will be updated
        self.change_context_window(context_window=context_window)

        if activation is None:
            activation = nn.ReLU(inplace=True)

        if PYTORCH_QUANTIZATION_AVAILABLE and quantize:
            self.fc = nn.Sequential(
                quant_nn.QuantLinear(channels, channels // reduction_ratio, bias=False),
                activation,
                quant_nn.QuantLinear(channels // reduction_ratio, channels, bias=False),
            )
        elif not PYTORCH_QUANTIZATION_AVAILABLE and quantize:
            raise ImportError(
                "pytorch-quantization is not installed. Install from "
                "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction_ratio, bias=False),
                activation,
                nn.Linear(channels // reduction_ratio, channels, bias=False),
            )
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # The use of negative indices on the transpose allow for expanded SqueezeExcite
        batch, channels, timesteps = x.size()[:3]
        # Computes in float32 to avoid instabilities during training with AMP.
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            if timesteps < self.context_window:
                y = self.gap(x)
            else:
                y = self.pool(x)  # [B, C, T - context_window + 1]
            y = y.transpose(1, -1)  # [B, T - context_window + 1, C]
            y = self.fc(y)  # [B, T - context_window + 1, C]
            y = y.transpose(1, -1)  # [B, C, T - context_window + 1]

        if self.context_window > 0:
            y = torch.nn.functional.interpolate(y, size=timesteps, mode=self.interpolation_mode)

        y = torch.sigmoid(y)
        return x * y

    def change_context_window(self, context_window: int):
        """
        Update the context window of the SqueezeExcitation module, in-place if possible.

        Will update the pooling layer to either nn.AdaptiveAvgPool1d() (for global SE) or nn.AvgPool1d()
        (for limited context SE).

        If only the context window is changing but still a limited SE context block - then
        the earlier instance of nn.AvgPool1d() will be updated.

        Args:
            context_window: An integer representing the number of input timeframes that will be used
                to compute the context. Each timeframe corresponds to a single window stride of the
                STFT features.

                Say the window_stride = 0.01s, then a context window of 128 represents 128 * 0.01 s
                of context to compute the Squeeze step.
        """
        if hasattr(self, 'context_window'):
            logging.info(f"Changing Squeeze-Excitation context window from {self.context_window} to {context_window}")

        self.context_window = int(context_window)

        if self.context_window <= 0:
            if PYTORCH_QUANTIZATION_AVAILABLE and self._quantize:
                if not isinstance(self.pool, quant_nn.QuantAdaptiveAvgPool1d(1)):
                    self.pool = quant_nn.QuantAdaptiveAvgPool1d(1)  # context window = T

            elif not PYTORCH_QUANTIZATION_AVAILABLE and self._quantize:
                raise ImportError(
                    "pytorch-quantization is not installed. Install from "
                    "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
                )

            else:
                if not isinstance(self.pool, nn.AdaptiveAvgPool1d):
                    self.pool = nn.AdaptiveAvgPool1d(1)  # context window = T
        else:
            if PYTORCH_QUANTIZATION_AVAILABLE and self._quantize:
                if not isinstance(self.pool, quant_nn.QuantAvgPool1d):
                    self.pool = quant_nn.QuantAvgPool1d(self.context_window, stride=1)

            elif not PYTORCH_QUANTIZATION_AVAILABLE and self._quantize:
                raise ImportError(
                    "pytorch-quantization is not installed. Install from "
                    "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
                )

            else:
                if not isinstance(self.pool, nn.AvgPool1d):
                    self.pool = nn.AvgPool1d(self.context_window, stride=1)
                else:
                    # update the context window
                    self.pool.kernel_size = _single(self.context_window)


class JasperBlock(nn.Module):
    """
    Constructs a single "Jasper" block. With modified parameters, also constructs other blocks for models
    such as `QuartzNet` and `Citrinet`.

    - For `Jasper`    : `separable` flag should be False
    - For `QuartzNet` : `separable` flag should be True
    - For `Citrinet`  : `separable` flag and `se` flag should be True

    Note that above are general distinctions, each model has intricate differences that expand over
    multiple such blocks.

    For further information about the differences between models which use JasperBlock, please review
    the configs for ASR models found in the ASR examples directory.

    Args:
        inplanes: Number of input channels.
        planes: Number of output channels.
        repeat: Number of repeated sub-blocks (R) for this block.
        kernel_size: Convolution kernel size across all repeated sub-blocks.
        kernel_size_factor: Floating point scale value that is multiplied with kernel size,
            then rounded down to nearest odd integer to compose the kernel size. Defaults to 1.0.
        stride: Stride of the convolutional layers.
        dilation: Integer which defined dilation factor of kernel. Note that when dilation > 1, stride must
            be equal to 1.
        padding: String representing type of padding. Currently only supports "same" padding,
            which symmetrically pads the input tensor with zeros.
        dropout: Floating point value, determins percentage of output that is zeroed out.
        activation: String representing activation functions. Valid activation functions are :
            {"hardtanh": nn.Hardtanh, "relu": nn.ReLU, "selu": nn.SELU, "swish": Swish}.
            Defaults to "relu".
        residual: Bool that determined whether a residual branch should be added or not.
            All residual branches are constructed using a pointwise convolution kernel, that may or may not
            perform strided convolution depending on the parameter `residual_mode`.
        groups: Number of groups for Grouped Convolutions. Defaults to 1.
        separable: Bool flag that describes whether Time-Channel depthwise separable convolution should be
            constructed, or ordinary convolution should be constructed.
        heads: Number of "heads" for the masked convolution. Defaults to -1, which disables it.
        normalization: String that represents type of normalization performed. Can be one of
            "batch", "group", "instance" or "layer" to compute BatchNorm1D, GroupNorm1D, InstanceNorm or
            LayerNorm (which are special cases of GroupNorm1D).
        norm_groups: Number of groups used for GroupNorm (if `normalization` == "group").
        residual_mode: String argument which describes whether the residual branch should be simply
            added ("add") or should first stride, then add ("stride_add"). Required when performing stride on
            parallel branch as well as utilizing residual add.
        residual_panes: Number of residual panes, used for Jasper-DR models. Please refer to the paper.
        conv_mask: Bool flag which determines whether to utilize masked convolutions or not. In general,
            it should be set to True.
        se: Bool flag that determines whether Squeeze-and-Excitation layer should be used.
        se_reduction_ratio: Integer value, which determines to what extend the hidden dimension of the SE
            intermediate step should be reduced. Larger values reduce number of parameters, but also limit
            the effectiveness of SE layers.
        se_context_window: Integer value determining the number of timesteps that should be utilized in order
            to compute the averaged context window. Defaults to -1, which means it uses global context - such
            that all timesteps are averaged. If any positive integer is used, it will utilize limited context
            window of that size.
        se_interpolation_mode: String used for interpolation mode of timestep dimension for SE blocks.
            Used only if context window is > 1.
            The modes available for resizing are: `nearest`, `linear` (3D-only),
            `bilinear`, `area`.
        stride_last: Bool flag that determines whether all repeated blocks should stride at once,
            (stride of S^R when this flag is False) or just the last repeated block should stride
            (stride of S when this flag is True).
        future_context: Int value that determins how many "right" / "future" context frames will be utilized
            when calculating the output of the conv kernel. All calculations are done for odd kernel sizes only.

            By default, this is -1, which is recomputed as the symmetric padding case.

            When future_context >= 0, will compute the asymmetric padding as follows :
            (left context, right context) = [K - 1 - future_context, future_context]

            Determining an exact formula to limit future context is dependent on global layout of the model.
            As such, we provide both "local" and "global" guidelines below.

            Local context limit (should always be enforced)
            - future context should be <= half the kernel size for any given layer
            - future context > kernel size defaults to symmetric kernel
            - future context of layer = number of future frames * width of each frame (dependent on stride)

            Global context limit (should be carefully considered)
            - future context should be layed out in an ever reducing pattern. Initial layers should restrict
            future context less than later layers, since shallow depth (and reduced stride) means each frame uses
            less amounts of future context.
            - Beyond a certain point, future context should remain static for a given stride level. This is
            the upper bound of the amount of future context that can be provided to the model on a global scale.
            - future context is calculated (roughly) as - (2 ^ stride) * (K // 2) number of future frames.
            This resultant value should be bound to some global maximum number of future seconds of audio (in ms).

            Note: In the special case where K < future_context, it is assumed that the kernel is too small to limit
            its future context, so symmetric padding is used instead.

            Note: There is no explicit limitation on the amount of future context used, as long as
            K > future_context constraint is maintained. This might lead to cases where future_context is
            more than half the actual kernel size K! In such cases, the conv layer is utilizing more of the future
            context than its current and past context to compute the output. While this is possible to do,
            it is not recommended and the layer will raise a warning to notify the user of such cases.
            It is advised to simply use symmetric padding for such cases.

            Example:
            Say we have a model that performs 8x stride and receives spectrogram frames with stride of 0.01s.
            Say we wish to upper bound future context to 80 ms.

            Layer ID, Kernel Size, Stride, Future Context, Global Context
            0, K=5,  S=1, FC=8, GC= 2 * (2^0) = 2 * 0.01 ms  (special case, K < FC so use symmetric pad)
            1, K=7,  S=1, FC=3, GC= 3 * (2^0) = 3 * 0.01 ms  (note that symmetric pad here uses 3 FC frames!)
            2, K=11, S=2, FC=4, GC= 4 * (2^1) = 8 * 0.01 ms  (note that symmetric pad here uses 5 FC frames!)
            3, K=15, S=1, FC=4, GC= 4 * (2^1) = 8 * 0.01 ms  (note that symmetric pad here uses 7 FC frames!)
            4, K=21, S=2, FC=2, GC= 2 * (2^2) = 8 * 0.01 ms  (note that symmetric pad here uses 10 FC frames!)
            5, K=25, S=2, FC=1, GC= 1 * (2^3) = 8 * 0.01 ms  (note that symmetric pad here uses 14 FC frames!)
            6, K=29, S=1, FC=1, GC= 1 * (2^3) = 8 * 0.01 ms ...
        quantize: Bool flag whether to quantize the Convolutional blocks.
    """

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
        se_context_window=-1,
        se_interpolation_mode='nearest',
        stride_last=False,
        future_context: int = -1,
        quantize=False,
    ):
        super(JasperBlock, self).__init__()

        if padding != "same":
            raise ValueError("currently only 'same' padding is supported")

        kernel_size_factor = float(kernel_size_factor)
        if type(kernel_size) in (list, tuple):
            kernel_size = [compute_new_kernel_size(k, kernel_size_factor) for k in kernel_size]
        else:
            kernel_size = compute_new_kernel_size(kernel_size, kernel_size_factor)

        if future_context < 0:
            padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])
        else:
            padding_val = get_asymtric_padding(kernel_size[0], stride[0], dilation[0], future_context)

        self.conv_mask = conv_mask
        self.separable = separable
        self.residual_mode = residual_mode
        self.se = se
        self.quantize = quantize

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
                    quantize=quantize,
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
                quantize=quantize,
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
                    quantize=quantize,
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
                        quantize=quantize,
                    )
                )

                res_list.append(res)

            self.res = res_list
            if PYTORCH_QUANTIZATION_AVAILABLE and self.quantize:
                self.residual_quantizer = quant_nn.TensorQuantizer(quant_nn.QuantConv2d.default_quant_desc_input)
            elif not PYTORCH_QUANTIZATION_AVAILABLE and quantize:
                raise ImportError(
                    "pytorch-quantization is not installed. Install from "
                    "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
                )
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
        quantize=False,
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
                quantize=quantize,
            )
        else:
            if PYTORCH_QUANTIZATION_AVAILABLE and quantize:
                return quant_nn.QuantConv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                    groups=groups,
                )
            elif not PYTORCH_QUANTIZATION_AVAILABLE and quantize:
                raise ImportError(
                    "pytorch-quantization is not installed. Install from "
                    "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
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
        quantize=False,
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
                    quantize=quantize,
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
                    quantize=quantize,
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
                    quantize=quantize,
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
        """
        Forward pass of the module.

        Args:
            input_: The input is a tuple of two values - the preprocessed audio signal as well as the lengths
                of the audio signal. The audio signal is padded to the shape [B, D, T] and the lengths are
                a torch vector of length B.

        Returns:
            The output of the block after processing the input through `repeat` number of sub-blocks,
            as well as the lengths of the encoded audio after padding/striding.
        """
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
                    if PYTORCH_QUANTIZATION_AVAILABLE and self.quantize:
                        out = self.residual_quantizer(out) + res_out
                    elif not PYTORCH_QUANTIZATION_AVAILABLE and self.quantize:
                        raise ImportError(
                            "pytorch-quantization is not installed. Install from "
                            "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
                        )
                    else:
                        out = out + res_out
                else:
                    out = torch.max(out, res_out)

        # compute the output
        out = self.mout(out)
        if self.res is not None and self.dense_residual:
            return xs + [out], lens

        return [out], lens
