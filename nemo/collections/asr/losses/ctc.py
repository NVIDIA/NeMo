# ! /usr/bin/python
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
from torch import nn

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType

__all__ = ['CTCLoss']


try:
    import k2

    K2_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    K2_AVAILABLE = False


def _transpose_input(log_probs, targets, input_lengths, target_lengths):
    

def resolve_ctc_loss(loss_name, blank_idx, reduction, zero_infinity, loss_kwargs):
    loss_kwargs = {} if loss_kwargs is None else loss_kwargs

    if loss_name == 'default':
        loss_func_base = nn.CTCLoss(blank=blank_idx, reduction=reduction, zero_infinity=zero_infinity)
        # here we transpose because we expect [B, T, D] while PyTorch assumes [T, B, D]
        def _transpose_input(log_probs, targets, input_lengths, target_lengths):
            return loss_func_base(log_probs.transpose(1, 0), targets, input_lengths, target_lengths)
        loss_func = _transpose_input
    elif loss_name == 'k2_fsa':
        if not K2_AVAILABLE:
            raise ImportError("k2 is not available")

        from nemo.collections.asr.parts.k2.ctc import CTCLoss
        # we assume that blank_idx + 1 == num_classes
        loss_func = CTCLoss(num_classes=blank_idx+1, blank=blank_idx, reduction=reduction, **loss_kwargs)
    else:
        raise ValueError(f"Invalid value of `loss_name`: {loss_name}.")
    return loss_func


class CTCLoss(Loss):
    @property
    def input_types(self):
        """Input types definitions for CTCLoss.
        """
        return {
            "log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "input_lengths": NeuralType(tuple('B'), LengthsType()),
            "target_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Output types definitions for CTCLoss.
        loss:
            NeuralType(None)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, num_classes, zero_infinity=False, reduction='mean_batch', loss_name="default", loss_kwargs=None):
        super(CTCLoss, self).__init__()

        self._blank = num_classes
        # Don't forget to properly call base constructor
        if reduction == 'mean_batch':
            ctc_reduction = 'none'
            self._apply_batch_mean = True
        elif reduction in ['sum', 'mean', 'none']:
            ctc_reduction = reduction
            self._apply_batch_mean = False
        self._loss = resolve_ctc_loss(
            loss_name=loss_name,
            blank_idx=self._blank,
            reduction=ctc_reduction,
            zero_infinity=zero_infinity,
            loss_kwargs=loss_kwargs
        )

    @typecheck()
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # override forward implementation
        # custom logic, if necessary
        input_lengths = input_lengths.long()
        target_lengths = target_lengths.long()
        targets = targets.long()
        loss = self._loss(
            log_probs=log_probs, targets=targets, input_lengths=input_lengths, target_lengths=target_lengths
        )
        if self._apply_batch_mean:
            loss = torch.mean(loss)
        return loss
