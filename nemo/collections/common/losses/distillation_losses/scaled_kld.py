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

import torch
import torch.nn as nn

from nemo.core.classes import Serialization, Typing, typecheck
from nemo.core.classes.mixins.distill_mixins import ScaledDistillationLossMixin
from nemo.core.neural_types import LogprobsType, LossType, NeuralType

__all__ = ['ScaledKLDivLoss']


class ScaledKLDivLoss(nn.KLDivLoss, Serialization, Typing, ScaledDistillationLossMixin):
    """
    Wrapper around the KLDivLoss so that gradients of the loss function are scaled by temperature parameter (T^2).
    Reference: Distilling the Knowledge in a Neural Network (https://arxiv.org/abs/1503.02531)
    """

    def __init__(
        self, temperature: float, size_average=None, reduce=None, reduction: str = 'batchmean', log_target: bool = False
    ):
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction, log_target=log_target)
        self.temperature = temperature

    @property
    def input_types(self):
        if self.log_target:
            return {
                "input": NeuralType(axes=None, elements_type=LogprobsType()),
                "target": NeuralType(axes=None, elements_type=LogprobsType()),
            }
        else:
            return None

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input=input, target=target)

    @property
    def grad_scale(self):
        """ Used to scale gradients of this loss function """
        return self.temperature ** 2
