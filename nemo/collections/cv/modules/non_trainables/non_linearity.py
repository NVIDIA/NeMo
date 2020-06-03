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


from typing import List, Optional

import torch

from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.core.neural_types import AxisKind, AxisType, LogprobsType, NeuralType, VoidType
from nemo.utils import logging
from nemo.utils.configuration_parsing import get_value_from_dictionary
from nemo.utils.decorators import add_port_docs

__all__ = ['NonLinearity']


class NonLinearity(NonTrainableNM):
    """
    Class responsible for applying additional non-linearity along the last axis of the input tensor.

    """

    def __init__(self, type: str = "logsoftmax", sizes: List[int] = [-1], name: Optional[str] = None):
        """
        Constructor initializing the  non-linearity.

        Args:
            type: Type of non-linearity (currently only logsoftmax is supported)
            sizes: Sizes of dimensions of the input/output tensors (DEFAULT: [-1] - variable size batch)
            name: Name of the module (DEFAULT: None)
        """
        # Call constructor of parent classes.
        NonTrainableNM.__init__(self, name=name)

        # Store params.
        self._type = type
        self._sizes = sizes

        # Get type - only one option accepted (for now).
        self._non_linearity_type = get_value_from_dictionary(type, ["logsoftmax"])

        # Apply the non-linearity along the last dimension.
        dim = len(sizes) - 1
        self._non_linearity = torch.nn.LogSoftmax(dim=dim)

    @property
    @add_port_docs()
    def input_ports(self):
        """
        Returns definitions of module input ports.
        Batch of inputs, each represented as index [BATCH_SIZE x ... x INPUT_SIZE]
        """
        # Prepare list of axes.
        axes = [AxisType(kind=AxisKind.Batch)]
        for size in self._sizes[1:]:
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
        for size in self._sizes[1:]:
            axes.append(AxisType(kind=AxisKind.Any, size=size))
        # Return neural type.
        # TODO: if self._type != "logsoftmax"
        return {"outputs": NeuralType(axes, LogprobsType())}

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
        # TODO: if self._type != "logsoftmax"
        return self._non_linearity(inputs)
