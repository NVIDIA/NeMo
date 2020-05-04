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

__all__ = ['ListwiseSoftmaxLoss']


class ListwiseSoftmaxLoss(LossNM):

    @property
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "scores": NeuralType(('B', 'D'), LogitsType()),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, list_size=2):
        LossNM.__init__(self)

        self._loss_fn = ListwiseSoftmax(list_size=list_size)

    def _loss_function(self, scores):
        loss = self._loss_fn(scores)
        return loss


class ListwiseSoftmax(torch.nn.Module):

    def __init__(self, list_size=2):
        super().__init__()
        self.list_size = list_size

    def forward(self, scores):
        log_probs = torch.log_softmax(scores.view(-1, self.list_size), dim=-1)
        neg_log_likelihood = -torch.mean(log_probs[:, 0])
        return neg_log_likelihood
