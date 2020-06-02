# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
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
# =============================================================================
# Copyright (C) IBM Corporation 2019
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
# =============================================================================

__author__ = "Younes Bouhadjar, Vincent Marois, Tomasz Kornuta"

"""
This file contains code artifacts adapted from the original implementation:
https://github.com/IBM/pytorchpipe/blob/develop/ptp/components/models/vision/convnet_encoder.py
"""


from typing import Optional

import numpy as np
import torch.nn as nn

from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_types import AxisKind, AxisType, ImageFeatureValue, ImageValue, NeuralType
from nemo.utils import logging
from nemo.utils.decorators import add_port_docs

__all__ = ['ConvNetEncoder']


class ConvNetEncoder(TrainableNM):
    """
    A simple image encoder consisting of 3 consecutive convolutional layers.
    The parameters of input image (height, width and depth) are not hardcoded
    so the encoder can be adjusted for a given application (image dimensions).
    """

    def __init__(
        self,
        input_depth: int,
        input_height: int,
        input_width: int,
        conv1_out_channels: int = 64,
        conv1_kernel_size: int = 3,
        conv1_stride: int = 1,
        conv1_padding: int = 0,
        maxpool1_kernel_size: int = 2,
        conv2_out_channels: int = 32,
        conv2_kernel_size: int = 3,
        conv2_stride: int = 1,
        conv2_padding: int = 0,
        maxpool2_kernel_size: int = 2,
        conv3_out_channels: int = 16,
        conv3_kernel_size: int = 3,
        conv3_stride: int = 1,
        conv3_padding: int = 0,
        maxpool3_kernel_size: int = 2,
        name: Optional[str] = None,
    ):
        """
        Constructor of the a simple CNN.

        The overall structure of this CNN is as follows:

            (Conv1 -> MaxPool1 -> ReLu) -> (Conv2 -> MaxPool2 -> ReLu) -> (Conv3 -> MaxPool3 -> ReLu)

        The parameters that the user can change are:

         - For Conv1, Conv2 & Conv3: number of output channels, kernel size, stride and padding.
         - For MaxPool1, MaxPool2 & MaxPool3: Kernel size


        .. note::

            We are using the default values of ``dilatation``, ``groups`` & ``bias`` for ``nn.Conv2D``.

            Similarly for the ``stride``, ``padding``, ``dilatation``, ``return_indices`` & ``ceil_mode`` of \
            ``nn.MaxPool2D``.

        Args: 
            input_depth: Depth of the input image
            input_height: Height of the input image
            input_width: Width of the input image
            convX_out_channels: Number of output channels of layer X (X=1,2,3)
            convX_kernel_size: Kernel size of layer X (X=1,2,3)
            convX_stride: Stride of layer X (X=1,2,3)
            convX_padding: Padding of layer X (X=1,2,3)
            name: Name of the module (DEFAULT: None)
        """
        # Call base constructor.
        TrainableNM.__init__(self, name=name)

        # Get input image information from the global parameters.
        self._input_depth = input_depth
        self._input_height = input_height
        self._input_width = input_width

        # Retrieve the Conv1 parameters.
        self._conv1_out_channels = conv1_out_channels
        self._conv1_kernel_size = conv1_kernel_size
        self._conv1_stride = conv1_stride
        self._conv1_padding = conv1_padding

        # Retrieve the MaxPool1 parameter.
        self._maxpool1_kernel_size = maxpool1_kernel_size

        # Retrieve the Conv2 parameters.
        self._conv2_out_channels = conv2_out_channels
        self._conv2_kernel_size = conv2_kernel_size
        self._conv2_stride = conv2_stride
        self._conv2_padding = conv2_padding

        # Retrieve the MaxPool2 parameter.
        self._maxpool2_kernel_size = maxpool2_kernel_size

        # Retrieve the Conv3 parameters.
        self._conv3_out_channels = conv3_out_channels
        self._conv3_kernel_size = conv3_kernel_size
        self._conv3_stride = conv3_stride
        self._conv3_padding = conv3_padding

        # Retrieve the MaxPool3 parameter.
        self._maxpool3_kernel_size = maxpool3_kernel_size

        # We can compute the spatial size of the output volume as a function of the input volume size (W),
        # the receptive field size of the Conv Layer neurons (F), the stride with which they are applied (S),
        # and the amount of zero padding used (P) on the border.
        # The corresponding equation is conv_size = ((W−F+2P)/S)+1.

        # doc for nn.Conv2D: https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
        # doc for nn.MaxPool2D: https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d

        # ----------------------------------------------------
        # Conv1
        self._conv1 = nn.Conv2d(
            in_channels=self._input_depth,
            out_channels=self._conv1_out_channels,
            kernel_size=self._conv1_kernel_size,
            stride=self._conv1_stride,
            padding=self._conv1_padding,
            dilation=1,
            groups=1,
            bias=True,
        )

        width_features_conv1 = np.floor(
            ((self._input_width - self._conv1_kernel_size + 2 * self._conv1_padding) / self._conv1_stride) + 1
        )
        height_features_conv1 = np.floor(
            ((self._input_height - self._conv1_kernel_size + 2 * self._conv1_padding) / self._conv1_stride) + 1
        )

        # ----------------------------------------------------
        # MaxPool1
        self._maxpool1 = nn.MaxPool2d(kernel_size=self._maxpool1_kernel_size)

        width_features_maxpool1 = np.floor(
            ((width_features_conv1 - self._maxpool1_kernel_size + 2 * self._maxpool1.padding) / self._maxpool1.stride)
            + 1
        )

        height_features_maxpool1 = np.floor(
            ((height_features_conv1 - self._maxpool1_kernel_size + 2 * self._maxpool1.padding) / self._maxpool1.stride)
            + 1
        )

        # ----------------------------------------------------
        # Conv2
        self._conv2 = nn.Conv2d(
            in_channels=self._conv1_out_channels,
            out_channels=self._conv2_out_channels,
            kernel_size=self._conv2_kernel_size,
            stride=self._conv2_stride,
            padding=self._conv2_padding,
            dilation=1,
            groups=1,
            bias=True,
        )

        width_features_conv2 = np.floor(
            ((width_features_maxpool1 - self._conv2_kernel_size + 2 * self._conv2_padding) / self._conv2_stride) + 1
        )
        height_features_conv2 = np.floor(
            ((height_features_maxpool1 - self._conv2_kernel_size + 2 * self._conv2_padding) / self._conv2_stride) + 1
        )

        # ----------------------------------------------------
        # MaxPool2
        self._maxpool2 = nn.MaxPool2d(kernel_size=self._maxpool2_kernel_size)

        width_features_maxpool2 = np.floor(
            ((width_features_conv2 - self._maxpool2_kernel_size + 2 * self._maxpool2.padding) / self._maxpool2.stride)
            + 1
        )
        height_features_maxpool2 = np.floor(
            ((height_features_conv2 - self._maxpool2_kernel_size + 2 * self._maxpool2.padding) / self._maxpool2.stride)
            + 1
        )

        # ----------------------------------------------------
        # Conv3
        self._conv3 = nn.Conv2d(
            in_channels=self._conv2_out_channels,
            out_channels=self._conv3_out_channels,
            kernel_size=self._conv3_kernel_size,
            stride=self._conv3_stride,
            padding=self._conv3_padding,
            dilation=1,
            groups=1,
            bias=True,
        )

        width_features_conv3 = np.floor(
            ((width_features_maxpool2 - self._conv3_kernel_size + 2 * self._conv3_padding) / self._conv3_stride) + 1
        )
        height_features_conv3 = np.floor(
            ((height_features_maxpool2 - self._conv3_kernel_size + 2 * self._conv3_padding) / self._conv3_stride) + 1
        )

        # ----------------------------------------------------
        # MaxPool3
        self._maxpool3 = nn.MaxPool2d(kernel_size=self._maxpool3_kernel_size)

        width_features_maxpool3 = np.floor(
            ((width_features_conv3 - self._maxpool3_kernel_size + 2 * self._maxpool3.padding) / self._maxpool3.stride)
            + 1
        )

        height_features_maxpool3 = np.floor(
            ((height_features_conv3 - self._maxpool1_kernel_size + 2 * self._maxpool3.padding) / self._maxpool3.stride)
            + 1
        )

        # Rememvber the output dims.
        self._feature_map_height = height_features_maxpool3
        self._feature_map_width = width_features_maxpool3
        self._feature_map_depth = self._conv3_out_channels

        # Log info about dimensions.
        logging.info('Input shape: [-1, {}, {}, {}]'.format(self._input_depth, self._input_height, self._input_width))
        logging.debug('Computed output shape of each layer:')
        logging.debug(
            '  * Conv1: [-1, {}, {}, {}]'.format(self._conv1_out_channels, height_features_conv1, width_features_conv1)
        )
        logging.debug(
            '  * MaxPool1: [-1, {}, {}, {}]'.format(
                self._conv1_out_channels, height_features_maxpool1, width_features_maxpool1
            )
        )
        logging.debug(
            '  * Conv2: [-1, {}, {}, {}]'.format(self._conv2_out_channels, height_features_conv2, width_features_conv2)
        )
        logging.debug(
            '  * MaxPool2: [-1, {}, {}, {}]'.format(
                self._conv2_out_channels, height_features_maxpool2, width_features_maxpool2
            )
        )
        logging.debug(
            '  * Conv3: [-1, {}, {}, {}]'.format(
                self._conv3_out_channels, height_features_conv3, width_features_conv3,
            )
        )
        logging.debug(
            '  * MaxPool3: [-1, {}, {}, {}]'.format(
                self._conv3_out_channels, width_features_maxpool3, height_features_maxpool3
            )
        )
        logging.info(
            'Output shape: [-1, {}, {}, {}]'.format(
                self._feature_map_depth, self._feature_map_height, self._feature_map_width
            )
        )

    @property
    @add_port_docs()
    def input_ports(self):
        """
        Returns definitions of module input ports.
        """
        return {
            "inputs": NeuralType(
                axes=(
                    AxisType(kind=AxisKind.Batch),
                    AxisType(kind=AxisKind.Channel, size=self._input_depth),
                    AxisType(kind=AxisKind.Height, size=self._input_height),
                    AxisType(kind=AxisKind.Width, size=self._input_width),
                ),
                elements_type=ImageValue(),
            )
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """
        Returns definitions of module output ports.
        """
        return {
            "outputs": NeuralType(
                axes=(
                    AxisType(kind=AxisKind.Batch),
                    AxisType(kind=AxisKind.Channel, size=self._feature_map_depth),
                    AxisType(kind=AxisKind.Height, size=self._feature_map_height),
                    AxisType(kind=AxisKind.Width, size=self._feature_map_width),
                ),
                elements_type=ImageFeatureValue(),
            )
        }

    def forward(self, inputs):
        """
        Forward pass of the convnet module.

        :param data_streams: DataStreams({'inputs','outputs'}), where:

        Args:
            inputs: Batch of inputs to be processed [BATCH_SIZE x INPUT_DEPTH x INPUT_HEIGHT x INPUT_WIDTH]
        
        Returns:
            Batch of outputs [BATCH_SIZE x OUTPUT_DEPTH x OUTPUT_HEIGHT x OUTPUT_WIDTH]

        """
        # apply Convolutional layer 1
        out_conv1 = self._conv1(inputs)

        # apply max_pooling and relu
        out_maxpool1 = nn.functional.relu(self._maxpool1(out_conv1))

        # apply Convolutional layer 2
        out_conv2 = self._conv2(out_maxpool1)

        # apply max_pooling and relu
        out_maxpool2 = nn.functional.relu(self._maxpool2(out_conv2))

        # apply Convolutional layer 3
        out_conv3 = self._conv3(out_maxpool2)

        # apply max_pooling and relu
        out_maxpool3 = nn.functional.relu(self._maxpool3(out_conv3))

        # Return output.
        return out_maxpool3
