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
# Copyright (C) tkornuta, IBM Corporation 2019
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

__author__ = "Tomasz Kornuta"

"""
This file contains code artifacts adapted from the original implementation:
https://github.com/IBM/pytorchpipe/blob/develop/ptp/components/transforms/reshape_tensor.py
"""

import torch

from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.core.neural_types import NeuralType, VoidType
from nemo.utils import logging
from nemo.utils.configuration_error import ConfigurationError
from nemo.utils.decorators import add_port_docs

__all__ = ['ReshapeTensor']


class ReshapeTensor(NonTrainableNM):
    """
    Class responsible for reshaping the input tensor.

    """

    def __init__(self, input_dims, output_dims, name=None):
        """
        Initializes object.

        """
        # Call constructor of parent classes.
        NonTrainableNM.__init__(self, name=name)

        # Get input and output shapes from configuration.
        self._input_dims = input_dims
        self._output_dims = output_dims

    @property
    @add_port_docs()
    def input_ports(self):
        """
        Returns definitions of module input ports.
        Batch of inputs, each represented as index [BATCH_SIZE x ... x INPUT_SIZE]
        """
        return {
            "inputs": NeuralType(['B'] + ['ANY'] * (len(self._input_dims) - 1), VoidType())
        }  # TODO: set proper sizes.

    @property
    @add_port_docs()
    def output_ports(self):
        """
        Returns definitions of module output ports.
        """
        return {
            "outputs": NeuralType(['B'] + ['ANY'] * (len(self._output_dims) - 1), VoidType())
        }  # TODO: set proper sizes of consecutive dimensions.

    def forward(self, inputs):
        """
        Encodes "inputs" in the format of a single tensor.
        Stores reshaped tensor in "outputs" field of in data_streams.

        Args:
            inputs: a tensor [BATCH_SIZE x ...]

        Returns:
            Outputs a tensor [BATCH_SIZE x ...] 
        """
        # print("{}: input shape: {}, device: {}\n".format(self.name, inputs.shape, inputs.device))

        # Reshape.
        return inputs.view(self._output_dims)
