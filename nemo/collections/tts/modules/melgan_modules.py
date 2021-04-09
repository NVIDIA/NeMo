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

#  The MIT License
#
#  Copyright (c) 2019 Tomoki Hayashi
#
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.

# The following functions/classes were based on code from https://github.com/kan-bayashi/ParallelWaveGAN/:
# ResidualStack
# MelGANGenerator
# MelGANDiscriminator
# MelGANMultiScaleDiscriminator

from typing import List

import numpy as np
import torch

from nemo.core.classes import NeuralModule
from nemo.core.classes.common import typecheck
from nemo.core.neural_types.elements import AudioSignal, MelSpectrogramType, VoidType
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging

__all__ = ['MelGANGenerator', 'MelGANDiscriminator', 'MelGANMultiScaleDiscriminator']


class ResidualStack(torch.nn.Module):
    """Residual stack module introduced in MelGAN."""

    def __init__(
        self,
        nonlinear_activation: torch.nn.Module,
        kernel_size: int,
        channels: int,
        dilation: int = 1,
        bias: bool = True,
    ):
        """Initialize ResidualStack module.
        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels of convolution layers.
            dilation (int): Dilation factor.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (torch.nn.Module): Activation function.
        """
        super().__init__()

        # defile residual stack part
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        self.stack = torch.nn.Sequential(
            nonlinear_activation,
            torch.nn.ReflectionPad1d((kernel_size - 1) // 2 * dilation),
            torch.nn.Conv1d(channels, channels, kernel_size, dilation=dilation, bias=bias),
            nonlinear_activation,
            torch.nn.Conv1d(channels, channels, 1, bias=bias),
        )

        # defile extra layer for skip connection
        self.skip_layer = torch.nn.Conv1d(channels, channels, 1, bias=bias)

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            c (Tensor): Input tensor (B, channels, T).
        Returns:
            Tensor: Output tensor (B, chennels, T).
        """
        return self.stack(x) + self.skip_layer(x)


class MelGANGenerator(NeuralModule):
    """MelGAN generator module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        channels: int,
        upsample_scales: List[int],
        stack_kernel_size: int,
        stacks: int,
        nonlinear_activation: torch.nn.Module = None,
        bias: bool = True,
        use_final_nonlinear_activation: bool = True,
        use_weight_norm: bool = True,
    ):
        """Initialize MelGANGenerator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            channels (int): Initial number of channels for conv layer.
            bias (bool): Whether to add bias parameter in convolution layers.
            upsample_scales (list): List of upsampling scales.
            stack_kernel_size (int): Kernel size of dilated conv layers in residual stack.
            stacks (int): Number of stacks in a single residual stack.
            nonlinear_activation (torch.nn.Module): Activation function.
            use_final_nonlinear_activation (bool): Whether to add an activation function for the final layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
        """
        super().__init__()

        # check hyper parameters is valid
        assert channels >= np.prod(upsample_scales)
        assert channels % (2 ** len(upsample_scales)) == 0
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."

        if nonlinear_activation is None:
            nonlinear_activation = torch.nn.LeakyReLU(negative_slope=0.2)

        # add initial layer
        layers = []
        layers += [
            torch.nn.ReflectionPad1d((kernel_size - 1) // 2),
            torch.nn.Conv1d(in_channels, channels, kernel_size, bias=bias),
        ]

        for i, upsample_scale in enumerate(upsample_scales):
            # add upsampling layer
            layers += [nonlinear_activation]
            layers += [
                torch.nn.ConvTranspose1d(
                    channels // (2 ** i),
                    channels // (2 ** (i + 1)),
                    upsample_scale * 2,
                    stride=upsample_scale,
                    padding=upsample_scale // 2 + upsample_scale % 2,
                    output_padding=upsample_scale % 2,
                    bias=bias,
                )
            ]

            # add residual stack
            for j in range(stacks):
                layers += [
                    ResidualStack(
                        kernel_size=stack_kernel_size,
                        channels=channels // (2 ** (i + 1)),
                        dilation=stack_kernel_size ** j,
                        bias=bias,
                        nonlinear_activation=nonlinear_activation,
                    )
                ]

        # add final layer
        layers += [nonlinear_activation]
        layers += [
            torch.nn.ReflectionPad1d((kernel_size - 1) // 2),
            torch.nn.Conv1d(channels // (2 ** len(upsample_scales)), out_channels, kernel_size, bias=bias),
        ]

        if use_final_nonlinear_activation:
            layers += [torch.nn.Tanh()]

        # define the model as a single function
        self.melgan = torch.nn.Sequential(*layers)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

    @property
    def input_types(self):
        return {
            "spec": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
        }

    @property
    def output_types(self):
        return {
            "audio": NeuralType(('B', 'S', 'T'), AudioSignal()),
        }

    @typecheck()
    def forward(self, *, spec):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, channels, T).

        Returns:
            Tensor: Output tensor (B, 1, T ** prod(upsample_scales)).

        """
        return self.melgan(spec)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(module):
            try:
                logging.debug(f"Weight norm is removed from {module}.")
                torch.nn.utils.remove_weight_norm(module)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(module):
            if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(module)
                logging.debug(f"Weight norm is applied to {module}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        """

        def _reset_parameters(module):
            if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.ConvTranspose1d):
                module.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {module}.")

        self.apply(_reset_parameters)


class MelGANDiscriminator(NeuralModule):
    """MelGAN discriminator module."""

    def __init__(
        self,
        kernel_sizes: int,
        channels: int,
        max_downsample_channels: int,
        downsample_scales: List[int],
        nonlinear_activation: torch.nn.Module,
        in_channels: int = 1,
        out_channels: int = 1,
        bias: bool = True,
    ):
        """Initilize MelGAN discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of two kernel sizes. The prod will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
                For example if kernel_sizes = [5, 3], the first layer kernel size will be 5 * 3 = 15,
                the last two layers' kernel size will be 5 and 3, respectively.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (torch.nn.Module): Activation function.
        """
        super().__init__()
        self.layers = torch.nn.ModuleList()

        # check kernel size is valid
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        # add first layer
        self.layers += [
            torch.nn.Sequential(
                torch.nn.ReflectionPad1d((np.prod(kernel_sizes) - 1) // 2),
                torch.nn.Conv1d(in_channels, channels, np.prod(kernel_sizes), bias=bias),
                nonlinear_activation,
            )
        ]

        # add downsample layers
        in_chs = channels
        for downsample_scale in downsample_scales:
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)
            self.layers += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chs,
                        out_chs,
                        kernel_size=downsample_scale * 10 + 1,
                        stride=downsample_scale,
                        padding=downsample_scale * 5,
                        groups=in_chs // 4,
                        bias=bias,
                    ),
                    nonlinear_activation,
                )
            ]
            in_chs = out_chs

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Conv1d(in_chs, out_chs, kernel_sizes[0], padding=(kernel_sizes[0] - 1) // 2, bias=bias,),
                nonlinear_activation,
            )
        ]
        self.layers += [
            torch.nn.Conv1d(out_chs, out_channels, kernel_sizes[1], padding=(kernel_sizes[1] - 1) // 2, bias=bias,),
        ]

    @property
    def input_types(self):
        return {
            "x": NeuralType(('B', 'S', 'T'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "decision": [NeuralType(('B', 'S', 'T'), VoidType())],
        }

    @typecheck()
    def forward(self, *, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of output tensors of each layer.

        """
        output = x
        for layer in self.layers:
            output = layer(output)

        return output


class MelGANMultiScaleDiscriminator(NeuralModule):
    """MelGAN multi-scale discriminator module."""

    def __init__(
        self,
        scales: int,
        downsample_scales: List[int],
        kernel_sizes: List[int],
        channels: int,
        nonlinear_activation: torch.nn.Module = None,
        in_channels: int = 1,
        out_channels: int = 1,
        max_downsample_channels: int = 1024,
        bias: bool = True,
        use_weight_norm: bool = True,
        use_spectral_norm: bool = False,
    ):
        """Initilize MelGAN multi-scale discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of two kernel sizes. The sum will be used for the first conv layer,
                and the first and the second kernel sizes will be used for the last two layers.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (torch.nn.Module): Activation function. If None, defaults to leaky ReLU
        """
        super().__init__()

        if nonlinear_activation is None:
            nonlinear_activation = torch.nn.LeakyReLU(negative_slope=0.2)
        self.discriminators = torch.nn.ModuleList()

        # add discriminators
        for _ in range(scales):
            self.discriminators += [
                MelGANDiscriminator(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes,
                    channels=channels,
                    max_downsample_channels=max_downsample_channels,
                    bias=bias,
                    downsample_scales=downsample_scales,
                    nonlinear_activation=nonlinear_activation,
                )
            ]
        self.pooling = torch.nn.AvgPool1d(kernel_size=4, stride=2, padding=1, count_include_pad=False)

        if use_spectral_norm and use_weight_norm:
            raise NotImplementedError

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()
            # reset parameters
            self.reset_parameters()

        if use_spectral_norm:
            self.apply_spectral_norm()

    @property
    def input_types(self):
        return {
            "x": NeuralType(('B', 'S', 'T'), AudioSignal()),
        }

    @property
    def output_types(self):
        return {
            "decision": [NeuralType((('B', 'S', 'T')), VoidType())],
        }

    @typecheck()
    def forward(self, *, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).

        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.

        """
        outs = []
        for disc in self.discriminators:
            outs += [disc(x=x)]
            x = self.pooling(x)

        return (outs,)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(module):
            try:
                logging.debug(f"Weight norm is removed from {module}.")
                torch.nn.utils.remove_weight_norm(module)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(module):
            if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(module)
                logging.debug(f"Weight norm is applied to {module}.")

        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_spectral_norm(module):
            if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.ConvTranspose1d):
                torch.nn.utils.spectral_norm(module)
                logging.debug(f"Weight norm is applied to {module}.")

        self.apply(_apply_spectral_norm)

    def reset_parameters(self):
        """Reset parameters.

        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py

        """

        def _reset_parameters(module):
            if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.ConvTranspose1d):
                module.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {module}.")

        self.apply(_reset_parameters)
