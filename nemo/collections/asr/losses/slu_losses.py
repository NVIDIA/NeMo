# ! /usr/bin/python
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.asr.parts.utils.slu_utils import get_seq_mask
from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType


def get_seq_masked_loss(
    loss_fn, predictions, targets, lengths=None, label_smoothing=0.0, reduction="mean",
):
    """
    Calculate the loss function specified by 'loss_fn' on the input predictions. 
    This function also supports label smoothing (https://arxiv.org/abs/1701.06548)

    params:
    - loss_fn: callable
    - predictions: BxTxC
    - targets: B
    - lengths: B
    """

    mask = torch.ones_like(predictions)
    if lengths is not None:
        mask = get_seq_mask(predictions, lengths).float()

    predictions = predictions.transpose(1, 2)  # BxTxC -> BxCxT

    loss = mask * loss_fn(predictions, targets)
    batch_size = loss.size(0)
    if reduction == "mean":
        loss = loss.sum() / torch.sum(mask)
    elif reduction == "batchmean":
        loss = loss.sum() / batch_size
    elif reduction == "batch":
        loss = loss.reshape(batch_size, -1).sum(1) / mask.reshape(batch_size, -1).sum(1)

    if label_smoothing == 0.0:
        return loss
    else:
        # Regularizing Neural Networks by Penalizing Confident Output Distributions.
        # https://arxiv.org/abs/1701.06548
        loss_reg = torch.mean(predictions, dim=1) * mask
        if reduction == "mean":
            loss_reg = torch.sum(loss_reg) / torch.sum(mask)
        elif reduction == "batchmean":
            loss_reg = torch.sum(loss_reg) / targets.shape[0]
        elif reduction == "batch":
            loss_reg = loss_reg.sum(1) / mask.sum(1)

        return -label_smoothing * loss_reg + (1 - label_smoothing) * loss


class SeqNLLLoss(Loss):
    def __init__(self, reduction='mean', label_smoothing=0.0, **kwargs):
        """
        Calculates the negative log likelihood (NLL) loss for sequence data,
        with optional sequence masking and label smoothing.
        """
        super().__init__()
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.nll_loss = nn.NLLLoss(reduction='none', **kwargs)

    @property
    def input_types(self):
        """
        Input types definitions for sequential NLL loss.
        """
        return {
            "log_probs": NeuralType(("B", "T", "D"), LogprobsType()),
            "targets": NeuralType(("B", "T"), LabelsType()),
            "lengths": NeuralType(tuple("B"), LengthsType(), optional=True),
        }

    @property
    def output_types(self):
        """O
        utput types definitions for sequential NLL loss.
        loss:
            NeuralType(LossType())
        """
        return {"loss": NeuralType(elements_type=LossType())}

    @typecheck()
    def forward(self, log_probs, targets, lengths=None):
        loss = get_seq_masked_loss(
            loss_fn=self.nll_loss,
            predictions=log_probs,
            targets=targets,
            lengths=lengths,
            label_smoothing=self.label_smoothing,
        )
        return loss
