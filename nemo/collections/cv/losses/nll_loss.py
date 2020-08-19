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

from typing import Optional

from torch.nn import NLLLoss as torch_NLLLoss

from nemo.core.classes import Serialization, Typing, typecheck
from nemo.core.neural_types import ClassificationTarget, LogprobsType, LossType, NeuralType
from nemo.utils.decorators import experimental


@experimental
class NLLLoss(torch_NLLLoss, Serialization, Typing):
    """ Class representing a simple NLL loss. """

    def __init__(self, name: Optional[str] = None):
        """
        Constructor.

        Args:
            name: Name of the module (DEFAULT: None)
        """
        # Call the base constructors.
        # Serialization.__init__(self, name=name)
        torch_NLLLoss.__init__(self)

    @property
    def input_types(self):
        """ Returns definitions of module input ports. """
        return {
            "predictions": NeuralType(axes=('B', 'ANY'), elements_type=LogprobsType()),
            "targets": NeuralType(axes=('B'), elements_type=ClassificationTarget()),
        }

    @property
    def output_types(self):
        """ Returns definitions of module output ports. """
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(self, predictions, targets):
        return torch_NLLLoss().forward(input=predictions, target=targets)
