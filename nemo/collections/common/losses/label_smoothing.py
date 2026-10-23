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

from typing import Optional, Sequence, Union

import torch

__all__ = ['label_smoothed_log_prob_gradient']


class _LabelSmoothedLogProbGradient(torch.autograd.Function):
    """Identity in the forward pass; smooths the gradient over the label dimension in the backward pass."""

    @staticmethod
    def forward(ctx, log_probs: torch.Tensor, factor, floor, keep, dim: int) -> torch.Tensor:
        ctx.factor = factor
        ctx.floor = floor
        ctx.keep = keep
        ctx.dim = dim
        return log_probs

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        # The incoming gradient is minus a (scaled) target distribution over the labels, so its absolute sum
        # is the mass that the smoothing redistributes; masked positions have a zero gradient and keep it.
        # With excluded labels, only the mass of the included labels is redistributed.
        mass = grad.abs() if ctx.keep is None else grad.abs() * ctx.keep
        total = mass.sum(dim=ctx.dim, keepdim=True)
        return grad * ctx.factor - ctx.floor * total, None, None, None, None


def label_smoothed_log_prob_gradient(
    log_probs: torch.Tensor,
    smoothing: Union[float, torch.Tensor],
    *,
    dim: int = -1,
    exclude_labels: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """
    Returns ``log_probs`` unchanged, with a gradient that corresponds to label smoothing.

    For a loss of the form ``-sum_i target[i] * log_probs[i]`` (cross-entropy, negative log-likelihood), the gradient
    with respect to ``log_probs`` is ``-target``. This function keeps the loss value, but replaces that gradient by
    the gradient of the same loss with the smoothed targets ``(1 - smoothing) * target + smoothing / (C - 1)`` for
    the other labels, ``C`` being the number of labels. The reported loss therefore stays the unsmoothed one, while
    the parameters receive the gradient of the label-smoothed loss.

    The gradient of the loss with respect to ``log_probs`` must be minus a (scaled) probability distribution over
    ``dim``, which holds for a negative log-likelihood on log-probabilities, also for a CTC or RNNT loss whose
    backward is taken with respect to the log-probabilities. ``torch.nn.functional.ctc_loss`` instead returns the
    gradient with respect to the logits behind the log-probabilities (``softmax - posterior``), so it is not
    compatible with this function.

    The idea and the form of the gradient transformation follow ``label_smoothed_log_prob_gradient`` of the RETURNN
    frontend (https://github.com/rwth-i6/returnn, ``returnn.frontend.label_smoothing``), which trains attention
    encoder-decoder models with label smoothing this way.

    Args:
        log_probs: log-probabilities of any shape with the label dimension ``dim``
        smoothing: label smoothing weight; a float or a scalar tensor (for a scheduled value)
        dim: the label dimension
        exclude_labels: labels that receive no smoothing mass and keep their gradient, such as a blank label; the
            mass of the other labels is then smoothed among those ``C - len(exclude_labels)`` labels, with the
            floor ``smoothing / (C - len(exclude_labels) - 1)``
    """
    if not isinstance(smoothing, torch.Tensor) and smoothing == 0:
        return log_probs
    num_labels = log_probs.shape[dim]
    keep = None
    if not exclude_labels:
        floor = smoothing / (num_labels - 1)
        factor = 1.0 - num_labels * floor
    else:
        excluded = sorted(set(int(label) for label in exclude_labels))
        if len(excluded) >= num_labels - 1:
            raise ValueError(f"Cannot exclude {len(excluded)} of {num_labels} labels from label smoothing")
        floor = smoothing / (num_labels - len(excluded) - 1)
        factor = 1.0 - (num_labels - len(excluded)) * floor
        keep = torch.ones(num_labels, dtype=log_probs.dtype, device=log_probs.device)
        keep[excluded] = 0.0
        shape = [1] * log_probs.dim()
        shape[dim] = num_labels
        keep = keep.view(shape)
        factor = keep * factor + (1.0 - keep)
        floor = keep * floor
    return _LabelSmoothedLogProbGradient.apply(log_probs, factor, floor, keep, dim)
