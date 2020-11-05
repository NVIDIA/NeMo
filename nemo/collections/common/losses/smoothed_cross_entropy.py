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

import torch

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types import LabelsType, LogprobsType, LossType, MaskType, NeuralType

__all__ = ["SmoothedCrossEntropyLoss"]


class SmoothedCrossEntropyLoss(Loss):
    """
    Calculates Cross-entropy loss with label smoothing for a batch of sequences.

    SmoothedCrossEntropyLoss:
    1) excludes padding tokens from loss calculation
    2) allows to use label smoothing regularization
    3) allows to calculate loss for the desired number of last tokens

    Args:
        label_smoothing (float): label smoothing regularization coefficient
        predict_last_k (int): parameter which sets the number of last tokens to calculate the loss for, for example
            0: (default) calculate loss on the entire sequence (e.g., NMT)
            1: calculate loss on the last token only (e.g., LM evaluation)
            Intermediate values allow to control the trade-off between eval
            time (proportional to the number of batches) and eval performance
            (proportional to the number of context tokens)
        pad_id (int): padding id
        eps (float): the small eps number to avoid division buy zero
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "log_probs": NeuralType(("B", "T", "D"), LogprobsType()),
            "labels": NeuralType(("B", "T"), LabelsType()),
            "output_mask": NeuralType(("B", "T"), MaskType(), optional=True),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(
        self,
        pad_id: Optional[int] = None,
        label_smoothing: Optional[float] = 0.0,
        predict_last_k: Optional[int] = 0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self._pad_id = pad_id
        self._eps = eps
        self._predict_last_k = predict_last_k
        self._label_smoothing = label_smoothing

    @typecheck()
    def forward(self, log_probs, labels, output_mask=None):
        """
        Args:
            log_probs: float tensor of shape batch_size x seq_len x vocab_size, values should be log probabilities
            labels: int tensor of shape batch_size x seq_len
            output_mask: binary tensor of shape batch_size x seq_len
            eps: epsilon param to avoid divide by zero in loss calculation
        """
        if output_mask is None and self._pad_id is None:
            raise ValueError("Both output_mask and pad_id are None")
        if output_mask is None and self._pad_id is not None:
            output_mask = (labels != self._pad_id).to(log_probs.dtype)

        if output_mask.dtype is not log_probs.dtype:
            output_mask = output_mask.to(log_probs.dtype)

        batch_size, seq_len, vocab_size = log_probs.size()
        smoothing = vocab_size * self._label_smoothing / (vocab_size - 1)
        target_log_probs = log_probs.gather(2, labels.unsqueeze(2)).squeeze(2)

        smoothing_log_probs = log_probs.mean(dim=-1)
        neg_log_likelihood = (1.0 - smoothing) * target_log_probs + smoothing * smoothing_log_probs
        neg_log_likelihood = neg_log_likelihood[:, -self._predict_last_k :]
        output_mask = output_mask[:, -self._predict_last_k :]
        neg_log_likelihood = -torch.sum(neg_log_likelihood * output_mask)
        neg_log_likelihood = neg_log_likelihood / (output_mask.sum() + self._eps)

        return neg_log_likelihood
