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

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F

from nemo.core import Loss, typecheck
from nemo.core.neural_types import EncodedRepresentation, LossType, NeuralType


class Wav2VecLoss(Loss):
    @property
    def input_types(self):
        """Input types definitions for Wav2VecLoss.
        """
        return {
            "logits": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "targets": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
            "negatives": NeuralType(('N', 'B', 'T', 'D'), EncodedRepresentation()),
            "prob_ppl_loss": NeuralType(elements_type=LossType()),
            "feature_loss": NeuralType(elements_type=LossType()),
        }

    @property
    def output_types(self):
        """Output types definitions for Wav2VecLoss.
        """
        return {
            "loss": NeuralType(elements_type=LossType()),
            "feature_loss": NeuralType(elements_type=LossType()),
            "prob_ppl_loss": NeuralType(elements_type=LossType()),
        }

    def __init__(self, feature_loss_weight: float, prob_ppl_weight: float, logit_temp: float, reduce: bool = True):
        """
        Compute the contrastive loss with respect to the model outputs and sampled negatives from quantizer codebooks.
        Args:
            feature_loss_weight: Feature penalty weight (L2 Norm)
            prob_ppl_weight: Perplexity Loss with respect to probabilities during quantization
            logit_temp: Temperature normalization applied in loss.
            reduce: Reduce loss via sum reduction (Default true)
        """
        super().__init__()
        self.feature_loss_weight = feature_loss_weight
        self.prob_ppl_weight = prob_ppl_weight
        self.logit_temp = logit_temp
        self.reduce = reduce

    @typecheck()
    def forward(
        self,
        logits: torch.tensor,
        targets: torch.tensor,
        negatives: torch.tensor,
        prob_ppl_loss: torch.tensor,
        feature_loss: torch.tensor,
    ) -> [torch.tensor, torch.tensor, torch.tensor]:
        """
        Args:
            logits: Model activations
            targets: The true target quantized representations
            negatives: Sampled negatives from the quantizer codebooks. Sampled from all other timesteps.
            feature_loss: Feature penalty (L2 Norm)
            prob_ppl_loss: Perplexity Loss with respect to probs in quantization
        Returns:
            output loss values, feature loss, prob_ppl loss (after scaling).
        """

        # Calculate similarity between logits and all targets, returning FxBxT
        similarity_scores = self._calculate_similarity(logits, negatives, targets)

        # Create targets of size B*T
        similarity_targets = logits.new_zeros(similarity_scores.size(1) * similarity_scores.size(2), dtype=torch.long)

        # Transpose similarity scores to (T*B)xF for loss
        similarity_scores = similarity_scores.transpose(0, 2)
        similarity_scores = similarity_scores.reshape(-1, similarity_scores.size(-1))

        loss = F.cross_entropy(similarity_scores, similarity_targets, reduction="sum" if self.reduce else "none")

        sample_size = similarity_targets.numel()

        if self.prob_ppl_weight != 0:
            prob_ppl_loss = self.prob_ppl_weight * prob_ppl_loss * sample_size
            loss += prob_ppl_loss

        if self.feature_loss_weight != 0:
            feature_loss = self.feature_loss_weight * feature_loss * sample_size
            loss += feature_loss

        return loss, feature_loss, prob_ppl_loss

    def _calculate_similarity(self, logits, negatives, targets):
        neg_is_pos = (targets == negatives).all(-1)
        targets = targets.unsqueeze(0)
        targets = torch.cat([targets, negatives], dim=0)
        logits = torch.cosine_similarity(logits.float(), targets.float(), dim=-1).type_as(logits)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        return logits
