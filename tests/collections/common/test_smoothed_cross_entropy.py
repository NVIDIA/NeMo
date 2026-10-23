# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import pytest
import torch

from nemo.collections.common.losses import SmoothedCrossEntropyLoss

NUM_LABELS = 7
SMOOTHING = 0.1


def smoothed_targets(labels: torch.Tensor, num_labels: int, smoothing: float) -> torch.Tensor:
    """The label-smoothed target distribution: ``1 - smoothing`` on the label, the rest spread evenly."""
    targets = torch.full(labels.shape + (num_labels,), smoothing / (num_labels - 1))
    return targets.scatter(-1, labels.unsqueeze(-1), 1.0 - smoothing)


class TestSmoothedCrossEntropyLossGradientOnly:
    @pytest.mark.unit
    def test_reports_the_plain_loss_with_the_smoothed_gradient(self):
        torch.manual_seed(1)
        logits = torch.randn(2, 4, NUM_LABELS)
        labels = torch.randint(NUM_LABELS, (2, 4))
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

        smoothed_loss = SmoothedCrossEntropyLoss(label_smoothing=SMOOTHING)
        gradient_only = SmoothedCrossEntropyLoss(label_smoothing=SMOOTHING, label_smoothing_gradient_only=True)
        plain_loss = SmoothedCrossEntropyLoss(label_smoothing=0.0)

        reference = logits.clone().requires_grad_()
        smoothed_loss(log_probs=reference.log_softmax(-1), labels=labels, output_mask=mask).backward()
        candidate = logits.clone().requires_grad_()
        value = gradient_only(log_probs=candidate.log_softmax(-1), labels=labels, output_mask=mask)
        value.backward()
        plain = plain_loss(log_probs=logits.log_softmax(-1), labels=labels, output_mask=mask)

        torch.testing.assert_close(value, plain)
        torch.testing.assert_close(candidate.grad, reference.grad)

    @pytest.mark.unit
    def test_per_token_losses_keep_the_smoothed_gradient(self):
        torch.manual_seed(2)
        logits = torch.randn(3, 2, NUM_LABELS)
        labels = torch.randint(NUM_LABELS, (3, 2))
        mask = torch.ones(3, 2)
        loss = SmoothedCrossEntropyLoss(
            label_smoothing=SMOOTHING, label_smoothing_gradient_only=True, per_token_reduction=False
        )
        candidate = logits.clone().requires_grad_()
        per_token = loss(log_probs=candidate.log_softmax(-1), labels=labels, output_mask=mask)
        assert per_token.shape == (3, 2)
        per_token.sum().backward()
        reference = logits.clone().requires_grad_()
        targets = smoothed_targets(labels, NUM_LABELS, SMOOTHING)
        (-(targets * reference.log_softmax(-1)).sum(-1)).sum().backward()
        torch.testing.assert_close(candidate.grad, reference.grad)
