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
https://github.com/IBM/pytorchpipe/blob/develop/ptp/components/models/general_usage/feed_forward_network.py
"""

from typing import List, Optional

import torch

from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_types import AxisKind, AxisType, NeuralType, VoidType
from nemo.utils import logging
from nemo.utils.configuration_error import ConfigurationError
from nemo.utils.decorators import add_port_docs

__all__ = ['FeedForwardNetwork']


class FeedForwardNetwork(TrainableNM):
    """
    A simple trainable module consisting of several stacked fully connected layers
    with ReLU non-linearities and dropout between them.
    
    # TODO: parametrize with other non-linearities.
    
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = [],
        dimensions: int = 2,
        dropout_rate: float = 0,
        name: Optional[str] = None,
    ):
        """
        Initializes the feed-forwad network.

        Args:
            input_size: Size of input (1D)
            output_sizes: Size of the output (1D)
            hidden_sizes: Sizes of the consecutive hidden layers (DEFAULT: [] = no hidden)
            dimensions: Number of dimensions of input/output tensors (DEFAULT: 2 = BATCH X INPUT_SIZE)
            dropout_rate: Dropout rage (Default: 0)
            name: Name of the module (DEFAULT: None)
        """
        # Call constructor of the parent class.
        TrainableNM.__init__(self, name=name)

        # Get input size.
        self._input_size = input_size
        if type(self._input_size) == list:
            if len(self._input_size) == 1:
                self._input_size = self._input_size[0]
            else:
                raise ConfigurationError("'input_size' must be a single value (received {})".format(self._input_size))

        # Get input/output dimensions, i.e. number of axes of the input [BATCH_SIZE x ... x INPUT_SIZE].
        # The module will "broadcast" over those dimensions.
        self._dimensions = dimensions
        if self._dimensions < 2:
            raise ConfigurationError("'dimensions' must be bigger than two  (received {})".format(self._dimensions))

        # Get output (prediction/logits) size.
        self._output_size = output_size
        if type(self._output_size) == list:
            if len(self._output_size) == 1:
                self._output_size = self._output_size[0]
            else:
                raise ConfigurationError(
                    "'output_size' must be a single value (received {})".format(self._output_size)
                )

        logging.info(
            "Initializing network with input size = {} and output size = {}".format(
                self._input_size, self._output_size
            )
        )

        # Create the module list.
        modules = []

        # Retrieve number of hidden layers, along with their sizes (numbers of hidden neurons from configuration).
        if type(hidden_sizes) == list:
            # Stack linear layers.
            input_dim = self._input_size
            for hidden_dim in hidden_sizes:
                # Add linear layer.
                modules.append(torch.nn.Linear(input_dim, hidden_dim))
                # Add activation.
                modules.append(torch.nn.ReLU())
                # Add dropout.
                if dropout_rate > 0:
                    modules.append(torch.nn.Dropout(dropout_rate))
                # Remember size.
                input_dim = hidden_dim

            # Add the last output" (or in a special case: the only) layer.
            modules.append(torch.nn.Linear(input_dim, self._output_size))

            logging.info("Created {} hidden layers with sizes {}".format(len(hidden_sizes), hidden_sizes))

        else:
            raise ConfigurationError(
                "'hidden_sizes' must contain a list with numbers of neurons in consecutive hidden layers (received {})".format(
                    hidden_sizes
                )
            )

        # Finally create the sequential model out of those modules.
        self.layers = torch.nn.Sequential(*modules)

    @property
    @add_port_docs()
    def input_ports(self):
        """
        Returns definitions of module input ports.
        Batch of inputs, each represented as index [BATCH_SIZE x ... x INPUT_SIZE]
        """
        # Prepare list of axes.
        axes = [AxisType(kind=AxisKind.Batch)]
        # Add the "additional dimensions".
        for _ in range(self._dimensions)[1:-1]:
            axes.append(AxisType(kind=AxisKind.Any))
        # Add the last axis: input_size
        axes.append(AxisType(kind=AxisKind.Any, size=self._input_size))
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
        # Add the "additional dimensions".
        for _ in range(self._dimensions)[1:-1]:
            axes.append(AxisType(kind=AxisKind.Any))
        # Add the last axis: input_size
        axes.append(AxisType(kind=AxisKind.Any, size=self._output_size))
        # Return neural type: batch of "logits" of "any type".
        return {"outputs": NeuralType(axes, VoidType())}

    def forward(self, inputs):
        """
        Performs the forward step of the module.

        Args:
            inputs: Batch of inputs to be processed [BATCH_SIZE x ... x INPUT_SIZE]
        
        Returns:
            Batch of outputs/predictions (log_probs) [BATCH_SIZE x ... x NUM_CLASSES]
        """

        # print("{}: input shape: {}, device: {}\n".format(self.name, inputs.shape, inputs.device))

        # Check that the input has the number of dimensions that we expect
        if len(inputs.shape) != self._dimensions:
            raise ConfigurationError(
                "Expected `{}` dimensions for input, but received `{}` instead. "
                F"Check fix the dimensions in your script.".format(self._dimensions, len(inputs.shape))
            )

        # Reshape such that we do a broadcast over the last dimension
        origin_shape = inputs.shape
        inputs = inputs.contiguous().view(-1, origin_shape[-1])

        # Propagate inputs through all layers and activations.
        outputs = self.layers(inputs)

        # Restore the input dimensions but the last one (as it's been resized by the FFN)
        outputs = outputs.view(*origin_shape[0 : self._dimensions - 1], -1)

        # Return the result.
        return outputs
