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

import torch

from nemo.backends.pytorch import LossNM
from nemo.collections.nlp.utils.data_utils import mask_padded_tokens
from nemo.core import LabelsType, LogitsType, LossType, MaskType, NeuralType

__all__ = ['HingeLoss']


class HingeLoss(LossNM):
    """
    Args:
        epsilon (float): default 0.2
    """

    @property
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "positive": NeuralType(('B', 'D'), LogitsType()),
            "negative": NeuralType(('B', 'D'), LogitsType()),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, epsilon=0.2):
        LossNM.__init__(self)

        self._loss_fn = Hinge(epsilon=epsilon)

    def _loss_function(self, positive, negative):
        loss = self._loss_fn(positive, negative)
        return loss


class Hinge(torch.nn.Module):

    def __init__(self, epsilon=0.2):
        super().__init__()
        self._eps = epsilon

    def forward(self, positive, negative):
        """
        Args:
            positive: float tensor of shape batch_size x 1, values should be logits
            negative: float tensor of shape batch_size x 1, values should be logits
        """

        loss = self._eps - (torch.sigmoid(positive) - torch.sigmoid(negative))
        loss = torch.max(loss, torch.zeros_like(loss)).mean()
        return loss
