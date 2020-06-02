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

from typing import List, Optional

from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.core.neural_types import AxisKind, AxisType, NeuralType, VoidType
from nemo.utils import logging
from nemo.utils.configuration_error import ConfigurationError
from nemo.utils.decorators import add_port_docs

__all__ = ['ReshapeTensor']


class ReshapeTensor(NonTrainableNM):
    """
    Class responsible for reshaping the input tensor.

    Reshapes tensor from e.g. [64, 16, 2, 2] to [64, 64].

    For more details please refer to: https://pytorch.org/docs/master/generated/torch.reshape.html
    """

    def __init__(self, input_sizes: List[int], output_sizes: List[int], name: Optional[str] = None):
        """
        Initializes the object.

        Args:
            input_sizes: Sizes of dimensions of the input tensor.
            output_sizes: Sizes of dimensions of the output.
            name: Name of the module (DEFAULT: None)
        """
        # Call constructor of parent classes.
        NonTrainableNM.__init__(self, name=name)

        # Validate params.
        if type(input_sizes) != list or len(input_sizes) < 2:
            raise ConfigurationError(
                "'input_sizes' must be at least a list with two values (received {})".format(self.input_sizes)
            )
        if type(output_sizes) != list or len(output_sizes) < 2:
            raise ConfigurationError(
                "'output_sizes' must be at least a list with two values (received {})".format(self.output_sizes)
            )

        # Get input and output shapes from configuration.
        self._input_sizes = input_sizes
        self._output_sizes = output_sizes

    @property
    @add_port_docs()
    def input_ports(self):
        """
        Returns definitions of module input ports.
        Batch of inputs, each represented as index [BATCH_SIZE x ... x INPUT_SIZE]
        """
        # Prepare list of axes.
        axes = [AxisType(kind=AxisKind.Batch)]
        for size in self._input_sizes[1:]:
            axes.append(AxisType(kind=AxisKind.Any, size=size))
        # Return neural type.
        return {"inputs": NeuralType(axes, VoidType())}

    @property
    @add_port_docs()
    def output_ports(self):
        """
        Returns definitions of module output ports.
        """
        # Prepare list of axes.
        axes = [AxisType(kind=AxisKind.Batch)]
        for size in self._output_sizes[1:]:
            axes.append(AxisType(kind=AxisKind.Any, size=size))
        # Return neural type.
        return {"outputs": NeuralType(axes, VoidType())}

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
        return inputs.view(self._output_sizes)
