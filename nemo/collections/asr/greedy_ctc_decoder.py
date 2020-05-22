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

from nemo.backends.pytorch.nm import NonTrainableNM
from nemo.core.neural_types import LogprobsType, NeuralType, PredictionsType
from nemo.utils.decorators import add_port_docs


class GreedyCTCDecoder(NonTrainableNM):
    """
    Greedy decoder that computes the argmax over a softmax distribution
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns:
            Definitions of module input ports.
        """
        return {"log_probs": NeuralType(('B', 'T', 'D'), LogprobsType())}

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns:
            Definitions of module output ports.
        """
        return {"predictions": NeuralType(('B', 'T'), PredictionsType())}

    def __init__(self):
        super().__init__()

    def forward(self, log_probs):
        argmx = log_probs.argmax(dim=-1, keepdim=False)
        return argmx
