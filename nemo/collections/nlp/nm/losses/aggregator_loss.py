# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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

from nemo.backends.pytorch import LossNM
from nemo.core import LossType, NeuralType
import torch

__all__ = ['LossAggregatorNM']


class LossAggregatorNM(LossNM):
    """
    Neural module which combines sums several losses into one.

    Args:
        num_inputs (int): number of input losses
    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.

        """
        input_ports = {}
        for i in range(self._num_losses):
            input_ports["loss_" + str(i + 1)] = NeuralType()

        return input_ports

    @property
    def output_ports(self):
        """Returns definitions of module output ports.

        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, num_inputs=2, weights=None):
        # Store number of inputs/losses.
        self._num_losses = num_inputs
        if weights is not None and len(weights) != num_inputs:
            raise("Len of weights should be equal to the number of inputs (num_inputs)")

        self._weights = weights
        LossNM.__init__(self)

    def _loss_function(self, **kwargs):
        values = [kwargs[x] for x in sorted(kwargs.keys())]
        loss = torch.zeros_like(values[0])
        for loss_idx, loss_value in enumerate(values):
            if self._weights is not None:
                loss = loss.add(loss_value, alpha=self._weight[loss_idx])
            else:
                loss = loss.add(loss_value)
        return loss
