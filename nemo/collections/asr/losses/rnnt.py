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

# Copyright 2018-2019, Mingkun Huang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types import LabelsType, LengthsType, LogprobsType, LossType, NeuralType

try:
    import warprnnt_pytorch as warprnnt

    WARP_RNNT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    WARP_RNNT_AVAILABLE = False


class RNNTLoss(Loss):
    @property
    def input_types(self):
        """Input types definitions for CTCLoss.
        """
        return {
            "log_probs": NeuralType(('B', 'T', 'D', 'D'), LogprobsType()),
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

    def __init__(self, num_classes, reduction='mean_batch'):
        """
        RNN-T Loss function based on https://github.com/HawkAaron/warp-transducer.

        Note:
            Requires the pytorch bindings to be installed prior to calling this class.

        Warning:
            In the case that GPU memory is exhausted in order to compute RNNTLoss, it might cause
            a core dump at the cuda level with the following error message.

            ```
                ...
                costs = costs.to(acts.device)
            RuntimeError: CUDA error: an illegal memory access was encountered
            terminate called after throwing an instance of 'c10::Error'
            ```

            Please kill all remaining python processes after this point, and use a smaller batch size
            for train, validation and test sets so that CUDA memory is not exhausted.

        Args:
            num_classes: Number of target classes for the joint network to predict.
                (Excluding the RNN-T blank token).

            reduction: Type of reduction to perform on loss. Possibly values are `mean`, `sum` or None.
                None will return a torch vector comprising the individual loss values of the batch.
        """
        super(RNNTLoss, self).__init__()

        if not WARP_RNNT_AVAILABLE:
            raise ImportError(
                "Could not import `warprnnt_pytorch`.\n"
                "Please visit https://github.com/HawkAaron/warp-transducer "
                "and follow the steps in the readme to build and install the "
                "pytorch bindings for RNNT Loss, or use the provided docker "
                "container that supports RNN-T loss."
            )

        if reduction not in [None, 'mean', 'sum', 'mean_batch']:
            raise ValueError('`reduction` must be one of [mean, sum, mean_batch]')

        self._blank = num_classes
        self.reduction = reduction
        self._loss = warprnnt.RNNTLoss(blank=self._blank, reduction='none')

    @typecheck()
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # Cast to int 32
        targets = targets.int()
        input_lengths = input_lengths.int()
        target_lengths = target_lengths.int()

        max_logit_len = input_lengths.max()
        max_targets_len = target_lengths.max()

        # Force cast joint to float32
        if log_probs.dtype != torch.float32:
            logits_orig = log_probs
            log_probs = log_probs.float()
            del logits_orig  # save memory *before* computing the loss

        # Ensure that shape mismatch does not occur due to padding
        # Due to padding and subsequent downsampling, it may be possible that
        # max sequence length computed does not match the actual max sequence length
        # of the log_probs tensor, therefore we increment the input_lengths by the difference.
        # This difference is generally small.
        if log_probs.shape[1] != max_logit_len:
            log_probs = log_probs.narrow(dim=1, start=0, length=max_logit_len).contiguous()

        # Reduce transcript length to correct alignment if additional padding was applied.
        # Transcript: [B, L] -> [B, L']; If L' < L
        if targets.shape[1] != max_targets_len:
            targets = targets.narrow(dim=1, start=0, length=max_targets_len)

        # Loss reduction can be dynamic, so set it prior to call
        if self.reduction != 'mean_batch':
            self._loss.reduction = self.reduction

        # Compute RNNT loss
        loss = self._loss(acts=log_probs, labels=targets, act_lens=input_lengths, label_lens=target_lengths)

        # Loss reduction can be dynamic, so reset it after call
        if self.reduction != 'mean_batch':
            self._loss.reduction = 'none'

        # Loss reduction only for mean_batch mode
        if self.reduction == 'mean_batch':
            loss = torch.mean(loss)

        # del new variables that may have been created
        del (
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )

        return loss
