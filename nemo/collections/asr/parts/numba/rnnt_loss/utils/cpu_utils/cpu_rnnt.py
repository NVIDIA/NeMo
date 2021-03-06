import multiprocessing

import torch
import numba
import math
from typing import Optional

from nemo.collections.asr.parts.numba.rnnt_loss.utils import global_constants


def log_sum_exp(a: torch.Tensor, b: torch.Tensor):
    if torch.isinf(a):
        return b

    if torch.isinf(b):
        return a

    if a > b:
        return math.log1p(math.exp(b - a)) + a
    else:
        return math.log1p(math.exp(a - b)) + b


class CpuRNNT_index:
    def __init__(self, U: int, maxU: int, minibatch: int, alphabet_size: int, batch_first: bool):
        super(CpuRNNT_index, self).__init__()
        self.U = U
        self.maxU = maxU
        self.minibatch = minibatch
        self.alphabet_size = alphabet_size
        self.batch_first = batch_first

    def __call__(self, t: int, u: int, v: Optional[int] = None):
        if v is None:
            return t * self.U + u
        else:
            if self.batch_first:
                return (t * self.maxU + u) * self.alphabet_size + v
            else:
                return (t * self.maxU + u) * self.minibatch * self.alphabet_size + v


class CpuRNNT_metadata:
    def __init__(
        self,
        T: int,
        U: int,
        workspace: torch.Tensor,
        bytes_used: int,
        blank: int,
        labels: torch.Tensor,
        log_probs: torch.Tensor,
        idx: CpuRNNT_index,
    ):
        super(CpuRNNT_metadata, self).__init__()

        self.alphas = workspace[bytes_used : bytes_used + T * U]
        bytes_used += T * U

        self.betas = workspace[bytes_used : bytes_used + T * U]
        bytes_used += T * U

        self.log_probs2 = workspace[bytes_used : bytes_used + T * U * 2]  # // only store blank & label
        bytes_used += T * U * 2

        self.bytes_used = bytes_used

        self.setup_probs(T, U, labels, blank, log_probs, idx)

    def setup_probs(
        self, T: int, U: int, labels: torch.Tensor, blank: int, log_probs: torch.Tensor, idx: CpuRNNT_index
    ):
        for t in range(T):
            for u in range(U):
                offset = (t * U + u) * 2
                self.log_probs2[offset] = log_probs[idx(t, u, blank)]
                # // labels do not have first blank
                if u < U - 1:
                    self.log_probs2[offset + 1] = log_probs[idx(t, u, labels[u])]


class CPURNNT:
    def __init__(
        self,
        minibatch: int,
        maxT: int,
        maxU: int,
        alphabet_size: int,
        workspace: torch.Tensor,
        blank: int,
        num_threads: int,
        batch_first: bool,
    ):
        self.minibatch_ = minibatch
        self.maxT_ = maxT
        self.maxU_ = maxU
        self.alphabet_size_ = alphabet_size
        self.workspace = workspace  # a flat vector of floatX numbers that represents allocated memory slices
        self.blank_ = blank
        self.num_threads_ = num_threads
        self.batch_first = batch_first

        if num_threads > 0:
            numba.set_num_threads(min(multiprocessing.cpu_count(), num_threads))
        else:
            self.num_threads_ = numba.get_num_threads()

    def cost_and_grad_kernel(
        self,
        log_probs: torch.Tensor,
        grad: torch.Tensor,
        labels: torch.Tensor,
        mb: int,
        T: int,
        U: int,
        bytes_used: int,
    ):
        idx = CpuRNNT_index(U, self.maxU_, self.minibatch_, self.alphabet_size_, self.batch_first)
        rnntm = CpuRNNT_metadata(T, U, self.workspace, bytes_used, self.blank_, labels, log_probs, idx)

        if self.batch_first:
            # zero grads
            grad *= 0.0

        llForward = self.compute_alphas(rnntm.log_probs2, T, U, rnntm.alphas)
        llBackward = self.compute_betas_and_grads(
            grad, rnntm.log_probs2, T, U, rnntm.alphas, rnntm.betas, labels, llForward
        )

        diff = (llForward - llBackward).abs()
        if diff > 0.1:
            print(f"WARNING: Forward backward likelihood mismatch : {diff}")

        return -llForward

    def compute_alphas(self, log_probs: torch.Tensor, T: int, U: int, alphas: torch.Tensor):
        idx = CpuRNNT_index(U, self.maxU_, self.minibatch_, self.alphabet_size_, self.batch_first)

        alphas[0] = 0
        for t in range(T):
            for u in range(U):
                if u == 0 and t > 0:
                    alphas[idx(t, 0)] = alphas[idx(t - 1, 0)] + log_probs[idx(t - 1, 0) * 2]

                if t == 0 and u > 0:
                    alphas[idx(0, u)] = alphas[idx(0, u - 1)] + log_probs[idx(0, u - 1) * 2 + 1]

                if t > 0 and u > 0:
                    no_emit = alphas[idx(t - 1, u)] + log_probs[idx(t - 1, u) * 2]
                    emit = alphas[idx(t, u - 1)] + log_probs[idx(t, u - 1) * 2 + 1]
                    alphas[idx(t, u)] = log_sum_exp(emit, no_emit)

        loglike = alphas[idx(T - 1, U - 1)] + log_probs[idx(T - 1, U - 1) * 2]
        return loglike

    def compute_betas_and_grads(
        self,
        grad: torch.Tensor,
        log_probs: torch.Tensor,
        T: int,
        U: int,
        alphas: torch.Tensor,
        betas: torch.Tensor,
        labels: torch.Tensor,
        logll: torch.Tensor,
    ):
        idx = CpuRNNT_index(U, self.maxU_, self.minibatch_, self.alphabet_size_, self.batch_first)
        betas[idx(T - 1, U - 1)] = log_probs[idx(T - 1, U - 1) * 2]

        for t in range(T - 1, -1, -1):
            for u in range(U - 1, -1, -1):
                if (u == U - 1) and (t < T - 1):
                    betas[idx(t, U - 1)] = betas[idx(t + 1, U - 1)] + log_probs[idx(t, U - 1) * 2]

                if (t == T - 1) and (u < U - 1):
                    betas[idx(T - 1, u)] = betas[idx(T - 1, u + 1)] + log_probs[idx(T - 1, u) * 2 + 1]

                if (t < T - 1) and (u < U - 1):
                    no_emit = betas[idx(t + 1, u)] + log_probs[idx(t, u) * 2]
                    emit = betas[idx(t, u + 1)] + log_probs[idx(t, u) * 2 + 1]
                    betas[idx(t, u)] = log_sum_exp(emit, no_emit)

        loglike = betas[0]
        # // Gradients w.r.t. log probabilities
        for t in range(T):
            for u in range(U):
                if t < T - 1:
                    g = alphas[idx(t, u)] + betas[idx(t + 1, u)]
                    grad[idx(t, u, self.blank_)] = -torch.exp(log_probs[idx(t, u) * 2] + g - loglike)

                if u < U - 1:
                    g = alphas[idx(t, u)] + betas[idx(t, u + 1)]
                    grad[idx(t, u, labels[u])] = -torch.exp(log_probs[idx(t, u) * 2 + 1] + g - loglike)

        # // gradient to the last blank transition
        grad[idx(T - 1, U - 1, self.blank_)] = -torch.exp(
            log_probs[idx(T - 1, U - 1) * 2] + alphas[idx(T - 1, U - 1)] - loglike
        )

        return loglike

    def cost_and_grad(
        self,
        log_probs: torch.Tensor,
        grads: torch.Tensor,
        costs: torch.Tensor,
        flat_labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> global_constants.RNNTStatus:
        # // per minibatch memory
        per_minibatch_bytes = 0

        # // alphas & betas
        per_minibatch_bytes += self.maxT_ * self.maxU_ * 2

        # // blank & label log probability cache
        per_minibatch_bytes += self.maxT_ * self.maxU_ * 2

        for mb in range(self.minibatch_):
            T = input_lengths[mb]  # // Length of utterance (time)
            U = label_lengths[mb] + 1  # // Number of labels in transcription
            batch_size = self.alphabet_size_
            if self.batch_first:
                batch_size = self.maxT_ * self.maxU_ * self.alphabet_size_

            costs[mb] = self.cost_and_grad_kernel(
                log_probs[(mb * batch_size) :],
                grads[(mb * batch_size) :],
                flat_labels[(mb * (self.maxU_ - 1)) :],
                mb,
                T,
                U,
                mb * per_minibatch_bytes,
            )

        return global_constants.RNNTStatus.RNNT_STATUS_SUCCESS

    def score_forward(
        self,
        log_probs: torch.Tensor,
        costs: torch.Tensor,
        flat_labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        # // per minibatch memory
        per_minibatch_bytes = 0

        # // alphas & betas
        per_minibatch_bytes += self.maxT_ * self.maxU_ * 2

        # // blank & label log probability cache
        per_minibatch_bytes += self.maxT_ * self.maxU_ * 2

        for mb in range(self.minibatch_):
            T = input_lengths[mb]  # // Length of utterance (time)
            U = label_lengths[mb] + 1  # // Number of labels in transcription
            batch_size = self.alphabet_size_
            if self.batch_first:
                batch_size = self.maxT_ * self.maxU_ * self.alphabet_size_

            idx = CpuRNNT_index(U, self.maxU_, self.minibatch_, self.alphabet_size_, self.batch_first)
            rnntm = CpuRNNT_metadata(
                T,
                U,
                self.workspace,
                mb * per_minibatch_bytes,
                self.blank_,
                flat_labels[(mb * (self.maxU_ - 1)) :],
                log_probs[(mb * batch_size) :],
                idx,
            )

            costs[mb] = -self.compute_alphas(rnntm.log_probs2, T, U, rnntm.alphas)

        return global_constants.RNNTStatus.RNNT_STATUS_SUCCESS
