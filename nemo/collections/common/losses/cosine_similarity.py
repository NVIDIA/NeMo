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

from torch import Tensor, nn

from nemo.core.classes import Serialization, Typing, typecheck
from nemo.core.neural_types import LossType, NeuralType

__all__ = ['CosineSimilarityLoss']


class CosineSimilarityLoss(nn.CosineSimilarity, Serialization, Typing):
    """
    CosineSimilarityLoss
    """

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, dim: int = 0, eps: float = 1e-8):
        """
        Compute the CosineSimilarity between two tensors (flattented to vectors),
        then utilize result in order to maximize similarity between the two inputs.

        .. math ::
            \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}
            \text{loss} = 1.0 - \text{similarity}

        Args:
            dim (int, optional): Dimension where cosine similarity is computed. Default: 0.
                Note: The default here is different from pytorch CosineSimilarity!
            eps (float, optional): Small value to avoid division by zero.
                Default: 1e-8

        Shape:
            - Input1: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`
            - Input2: :math:`(\ast_1, D, \ast_2)`, same shape as the Input1
            - Output: :math:`(\ast_1, \ast_2)`
        """
        super().__init__(dim=dim, eps=eps)

    @typecheck()
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        """
        Args:
            x1: Any tensor
            x2: Any tensor
        """
        x1 = x1.view(-1)
        x2 = x2.view(-1)
        cossine_similarity = super().forward(x1, x2)
        loss = 1.0 - cossine_similarity
        return loss
