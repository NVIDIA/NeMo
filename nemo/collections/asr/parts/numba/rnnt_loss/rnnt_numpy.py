# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#
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


import numpy as np
import torch
from torch.autograd import Function, Variable
from torch.nn import Module


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, (
        "gradients only computed for log_probs - please " "mark other tensors as not requiring gradients"
    )


def forward_pass(log_probs, labels, blank):
    """
    Computes probability of the forward variable alpha.

    Args:
        log_probs: Tensor of shape [T, U, V+1]
        labels: Labels of shape [B, U]
        blank: Index of the blank token.

    Returns:
        A tuple of the forward variable probabilities - alpha of shape [T, U]
        and the log likelihood of this forward step.
    """
    T, U, _ = log_probs.shape
    alphas = np.zeros((T, U), dtype='f')

    for t in range(1, T):
        alphas[t, 0] = alphas[t - 1, 0] + log_probs[t - 1, 0, blank]

    for u in range(1, U):
        alphas[0, u] = alphas[0, u - 1] + log_probs[0, u - 1, labels[u - 1]]
    for t in range(1, T):
        for u in range(1, U):
            no_emit = alphas[t - 1, u] + log_probs[t - 1, u, blank]
            emit = alphas[t, u - 1] + log_probs[t, u - 1, labels[u - 1]]
            alphas[t, u] = np.logaddexp(emit, no_emit)

    loglike = alphas[T - 1, U - 1] + log_probs[T - 1, U - 1, blank]
    return alphas, loglike


def backward_pass(log_probs, labels, blank):
    """
    Computes probability of the backward variable beta.

    Args:
        log_probs: Tensor of shape [T, U, V+1]
        labels: Labels of shape [B, U]
        blank: Index of the blank token.

    Returns:
        A tuple of the backward variable probabilities - beta of shape [T, U]
        and the log likelihood of this backward step.
    """
    T, U, _ = log_probs.shape
    betas = np.zeros((T, U), dtype='f')
    betas[T - 1, U - 1] = log_probs[T - 1, U - 1, blank]

    for t in reversed(range(T - 1)):
        betas[t, U - 1] = betas[t + 1, U - 1] + log_probs[t, U - 1, blank]

    for u in reversed(range(U - 1)):
        betas[T - 1, u] = betas[T - 1, u + 1] + log_probs[T - 1, u, labels[u]]

    for t in reversed(range(T - 1)):
        for u in reversed(range(U - 1)):
            no_emit = betas[t + 1, u] + log_probs[t, u, blank]
            emit = betas[t, u + 1] + log_probs[t, u, labels[u]]
            betas[t, u] = np.logaddexp(emit, no_emit)

    return betas, betas[0, 0]


def compute_gradient(log_probs, alphas, betas, labels, blank):
    """
    Computes the gradients of the log_probs with respect to the log probability of this step occuring.

    Args:
    Args:
        log_probs: Tensor of shape [T, U, V+1]
        alphas: Tensor of shape [T, U] which represents the forward variable.
        betas: Tensor of shape [T, U] which represents the backward variable.
        labels: Labels of shape [B, U]
        blank: Index of the blank token.

    Returns:
        Gradients of shape [T, U, V+1] with respect to the forward log probability
    """
    T, U, _ = log_probs.shape
    grads = np.full(log_probs.shape, -float("inf"))
    log_like = betas[0, 0]

    # // grad to last blank transition
    grads[T - 1, U - 1, blank] = alphas[T - 1, U - 1]

    grads[: T - 1, :, blank] = alphas[: T - 1, :] + betas[1:, :]
    for u, l in enumerate(labels):
        grads[:, u, l] = alphas[:, u] + betas[:, u + 1]

    grads = -np.exp(grads + log_probs - log_like)
    return grads


def transduce(log_probs, labels, blank=0):
    """
    Args:
        log_probs: 3D array with shape
              [input len, output len + 1, vocab size]
        labels: 1D array with shape [output time steps]
    Returns:
        float: The negative log-likelihood
        3D array: Gradients with respect to the
                    unnormalized input actications
    """
    alphas, ll_forward = forward_pass(log_probs, labels, blank)
    betas, ll_backward = backward_pass(log_probs, labels, blank)
    grads = compute_gradient(log_probs, alphas, betas, labels, blank)
    return -ll_forward, grads


def transduce_batch(log_probs, labels, flen, glen, blank=0):
    """
    Compute the transducer loss of the batch.

    Args:
        log_probs: [B, T, U, V+1]. Activation matrix normalized with log-softmax.
        labels: [B, U+1] - ground truth labels with <SOS> padded as blank token in the beginning.
        flen: Length vector of the acoustic sequence.
        glen: Length vector of the target sequence.
        blank: Id of the blank token.

    Returns:
        Batch of transducer forward log probabilities (loss) and the gradients of the activation matrix.
    """
    grads = np.zeros_like(log_probs)
    costs = []
    for b in range(log_probs.shape[0]):
        t = int(flen[b])
        u = int(glen[b]) + 1
        ll, g = transduce(log_probs[b, :t, :u, :], labels[b, : u - 1], blank)
        grads[b, :t, :u, :] = g
        costs.append(ll)
    return costs, grads


class _RNNT(Function):
    @staticmethod
    def forward(ctx, acts, labels, act_lens, label_lens, blank):
        costs, grads = transduce_batch(
            acts.detach().cpu().numpy(), labels.cpu().numpy(), act_lens.cpu().numpy(), label_lens.cpu().numpy(), blank,
        )

        costs = torch.FloatTensor([sum(costs)])
        grads = torch.Tensor(grads).to(acts)

        ctx.grads = Variable(grads)
        return costs

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.grads, None, None, None, None


class RNNTLoss(Module):
    """
    Parameters:
        `blank_label` (int): default 0 - label index of blank token
    """

    def __init__(self, blank: int = 0):
        super(RNNTLoss, self).__init__()
        self.blank = blank
        self.rnnt = _RNNT.apply

    def forward(self, acts, labels, act_lens, label_lens):
        assert len(labels.size()) == 2
        _assert_no_grad(labels)
        _assert_no_grad(act_lens)
        _assert_no_grad(label_lens)

        acts = torch.nn.functional.log_softmax(acts, -1)
        return self.rnnt(acts, labels, act_lens, label_lens, self.blank)
