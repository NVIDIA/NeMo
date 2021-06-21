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
from nemo.core.neural_types import LossType, NeuralType

__all__ = ['CosineEmbeddingLossWrapper']


class CosineEmbeddingLossWrapper(nn.CosineEmbeddingLoss, Serialization, Typing):
    """
    Wrapper around the CosineEmbeddingLoss so that it acts as a binary argument loss function.
    The third parameter - `target` is enforced as a torch.tensor(1) which is cached on the input tensor's device.
    """

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Any tensor
            x2: Any tensor
        """
        x1 = x1.view(1, -1)
        x2 = x2.view(1, -1)

        # Cache the value of _target
        if hasattr(self, '_target'):
            target = self._target

            if target.device != x1.device or target.dtype != x1.dtype:
                target = target.to(device=x1.device, dtype=x1.dtype)
        else:
            target = torch.tensor(1.0, device=x1.device, dtype=x1.dtype)
            self._target = target

        return super().forward(x1, x2, target)
