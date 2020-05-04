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
from nemo.core import LabelsType, LogitsType, LossType, MaskType, NeuralType, ChannelType

__all__ = ['DensePassageRetrievalLoss']


class DensePassageRetrievalLoss(LossNM):

    @property
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "queries": NeuralType(('B', 'T', 'D'), ChannelType()),
            "passages": NeuralType(('B', 'T', 'D'), ChannelType()),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            "scores": NeuralType(('B', 'B'), ChannelType()),
            "loss": NeuralType(elements_type=LossType()),
        }

    def __init__(self, num_negatives=1):
        LossNM.__init__(self)

        self._loss_fn = DPRLoss(num_negatives=num_negatives)

    def _loss_function(self, queries, passages):
        scores, loss = self._loss_fn(queries, passages)
        return scores, loss


class DPRLoss(torch.nn.Module):

    def __init__(self, num_negatives=2):
        super().__init__()
        self.num_negatives = num_negatives

    def forward(self, queries, passages):
        
        q_vectors = queries[:, 0, :] # Tensor of shape B x H
        p_vectors = passages[:, 0, :].view(-1, self.num_negatives + 1, passages.shape[-1]) # Tensor of shape B x d x H
        
        p_positives, p_negatives = p_vectors[:, 0], p_vectors[:, 1:]
        
        scores = torch.cat(
            (torch.matmul(q_vectors, p_positives.T),
            torch.einsum("ij,ipj->ip", q_vectors, p_negatives)),
            dim=1
        ) # Tensor of shape B x (B + d - 1)
        log_probs = torch.log_softmax(scores, dim=-1)
        neg_log_likelihood = -torch.mean(log_probs.diag())
        return scores, neg_log_likelihood
