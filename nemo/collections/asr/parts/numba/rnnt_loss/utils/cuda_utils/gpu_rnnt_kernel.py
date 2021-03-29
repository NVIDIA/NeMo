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

import math

import torch
from numba import cuda

from nemo.collections.asr.parts.numba.rnnt_loss.utils import rnnt_helper

GPU_RNNT_THREAD_SIZE = 128


@cuda.jit(device=True, inline=True)
def logp(
    denom: torch.Tensor, acts: torch.Tensor, maxT: int, maxU: int, alphabet_size: int, mb: int, t: int, u: int, v: int
):
    """
    Compute the sum of log probability from the activation tensor and its denominator.

    Args:
        denom: Tensor of shape [B, T, U] flattened. Represents the denominator of the logprobs activation tensor
            across entire vocabulary.
        acts: Tensor of shape [B, T, U, V+1] flattened. Represents the logprobs activation tensor.
        maxT: The maximum possible acoustic sequence length. Represents T in the logprobs tensor.
        maxU: The maximum possible target sequence length. Represents U in the logprobs tensor.
        alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT blank).
        mb: Batch indexer.
        t: Acoustic sequence timestep indexer.
        u: Target sequence timestep indexer.
        v: Vocabulary token indexer.

    Returns:
        The sum of logprobs[mb, t, u, v] + denom[mb, t, u]
    """
    col = (mb * maxT + t) * maxU + u
    return denom[col] + acts[col * alphabet_size + v]


@cuda.jit()
def compute_alphas_kernel(
    acts: torch.Tensor,
    denom: torch.Tensor,
    alphas: torch.Tensor,
    llForward: torch.Tensor,
    xlen: torch.Tensor,
    ylen: torch.Tensor,
    mlabels: torch.Tensor,  # [B]
    minibatch: int,
    maxT: int,
    maxU: int,
    alphabet_size: int,
    blank_: int,
):
    """
    Compute alpha (forward variable) probabilities over the transduction step.

    Args:
        acts: Tensor of shape [B, T, U, V+1] flattened. Represents the logprobs activation tensor.
        denom: Tensor of shape [B, T, U] flattened. Represents the denominator of the logprobs activation tensor
            across entire vocabulary.
        alphas: Zero tensor of shape [B, T, U]. Will be updated inside the kernel with the forward variable
            probabilities.
        llForward: Zero tensor of shape [B]. Represents the log-likelihood of the forward pass.
            Returned as the forward pass loss that is reduced by the optimizer.
        xlen: Vector of length B which contains the actual acoustic sequence lengths in the padded
            activation tensor.
        ylen: Vector of length B which contains the actual target sequence lengths in the padded
            activation tensor.
        mlabels: Matrix of shape [B, U+1] (+1 here is due to <SOS> token - usually the RNNT blank).
            The matrix contains the padded target transcription that must be predicted.
        minibatch: Int representing the batch size.
        maxT: The maximum possible acoustic sequence length. Represents T in the logprobs tensor.
        maxU: The maximum possible target sequence length. Represents U in the logprobs tensor.
        alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT blank).
        blank_: Index of the RNNT blank token in the vocabulary. Generally the first or last token in the vocab.

    Updates:
        Kernel inplace updates the following inputs:
        -   alphas: forward variable scores.
        -   llForward: log-likelihood of forward variable.
    """
    # // launch B blocks, each block has U threads
    b = cuda.blockIdx.x  # // batch id
    u = cuda.threadIdx.x  # label id, u
    T = xlen[b]  # select AM length of current sample
    U = ylen[b] + 1  # select target length of current sample, +1 for the blank token

    labels: torch.Tensor = mlabels[b]  # mb label start point, equivalent to mlabels + b * (maxU - 1)
    offset = b * maxT * maxU  # pointer indexing offset

    # alphas += offset # pointer offset, ignored since we explicitly add offset

    # Initilize alpha[b, t=0, u=0] for all b in B
    if u == 0:
        alphas[offset] = 0

    # sync until all alphas are initialized
    cuda.syncthreads()

    # Ordinary alpha calculations, broadcast across B=b and U=u
    # Look up forward variable calculation from rnnt_numpy.forward_pass()
    for n in range(1, T + U - 1):
        t = n - u

        if u == 0:
            # for t in range(1, T) step to initialize alphas[b, t, 0]
            if t > 0 and t < T:
                alphas[offset + t * maxU + u] = alphas[offset + (t - 1) * maxU + u] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, t - 1, 0, blank_
                )
        elif u < U:
            # for u in range(1, U) step to initialize alphas[b, 0, u]
            if t == 0:
                alphas[offset + u] = alphas[offset + u - 1] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, 0, u - 1, labels[u - 1]
                )

            # for t in range(1, T) for u in range(1, U) step to compute alphas[b, t, u]
            elif t > 0 and t < T:
                no_emit = alphas[offset + (t - 1) * maxU + u] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, t - 1, u, blank_
                )
                emit = alphas[offset + t * maxU + u - 1] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, t, u - 1, labels[u - 1]
                )

                alphas[offset + t * maxU + u] = rnnt_helper.log_sum_exp(emit, no_emit)

        # sync across all B=b and U=u
        cuda.syncthreads()

    # After final sync, alphas[b, T-1, U - 1] + logprobs[b, T-1, U-1, blank] + denom[b, T-1, U-1] gives
    # log-likelihood of forward pass.
    if u == 0:
        loglike = alphas[offset + (T - 1) * maxU + U - 1] + logp(
            denom, acts, maxT, maxU, alphabet_size, b, T - 1, U - 1, blank_
        )
        llForward[b] = loglike


@cuda.jit()
def compute_betas_kernel(
    acts: torch.Tensor,
    denom: torch.Tensor,
    betas: torch.Tensor,
    llBackward: torch.Tensor,
    xlen: torch.Tensor,
    ylen: torch.Tensor,
    mlabels: torch.Tensor,  # [B, U]
    minibatch: int,
    maxT: int,
    maxU: int,
    alphabet_size: int,
    blank_: int,
):
    """
    Compute beta (backward variable) probabilities over the transduction step.

    Args:
        acts: Tensor of shape [B, T, U, V+1] flattened. Represents the logprobs activation tensor.
        denom: Tensor of shape [B, T, U] flattened. Represents the denominator of the logprobs activation tensor
            across entire vocabulary.
        betas: Zero tensor of shape [B, T, U]. Will be updated inside the kernel with the backward variable
            probabilities.
        llBackward: Zero tensor of shape [B]. Represents the log-likelihood of the backward pass.
            Returned as the backward pass loss that is reduced by the optimizer.
        xlen: Vector of length B which contains the actual acoustic sequence lengths in the padded
            activation tensor.
        ylen: Vector of length B which contains the actual target sequence lengths in the padded
            activation tensor.
        mlabels: Matrix of shape [B, U+1] (+1 here is due to <SOS> token - usually the RNNT blank).
            The matrix contains the padded target transcription that must be predicted.
        minibatch: Int representing the batch size.
        maxT: The maximum possible acoustic sequence length. Represents T in the logprobs tensor.
        maxU: The maximum possible target sequence length. Represents U in the logprobs tensor.
        alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT blank).
        blank_: Index of the RNNT blank token in the vocabulary. Generally the first or last token in the vocab.

    Updates:
        Kernel inplace updates the following inputs:
        -   betas: backward variable scores.
        -   llBackward: log-likelihood of backward variable.
    """
    # // launch B blocks, each block has U threads
    b = cuda.blockIdx.x  # // batch id
    u = cuda.threadIdx.x  # label id, u
    T = xlen[b]  # select AM length of current sample
    U = ylen[b] + 1  # select target length of current sample, +1 for the blank token

    labels: torch.Tensor = mlabels[b]  # mb label start point, equivalent to mlabels + b * (maxU - 1)
    offset = b * maxT * maxU  # pointer indexing offset

    # betas += offset # pointer offset, ignored since we explicitly add offset

    # Initilize beta[b, t=T-1, u=U-1] for all b in B with log_probs[b, t=T-1, u=U-1, blank]
    if u == 0:
        betas[offset + (T - 1) * maxU + U - 1] = logp(denom, acts, maxT, maxU, alphabet_size, b, T - 1, U - 1, blank_)

    # sync until all betas are initialized
    cuda.syncthreads()

    # Ordinary beta calculations, broadcast across B=b and U=u
    # Look up backward variable calculation from rnnt_numpy.backward_pass()
    for n in range(T + U - 2, -1, -1):
        t = n - u

        if u == (U - 1):
            # for t in reversed(range(T - 1)) step to initialize betas[b, t, U-1]
            if t >= 0 and t < (T - 1):
                betas[offset + t * maxU + U - 1] = betas[offset + (t + 1) * maxU + U - 1] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, t, U - 1, blank_
                )
        elif u < U:
            if t == T - 1:
                # for u in reversed(range(U - 1)) step to initialize betas[b, T-1, u]
                betas[offset + (T - 1) * maxU + u] = betas[offset + (T - 1) * maxU + u + 1] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, T - 1, u, labels[u]
                )
            elif (t >= 0) and (t < T - 1):
                # for t in reversed(range(T - 1)) for u in reversed(range(U - 1)) step to compute betas[b, t, u]
                no_emit = betas[offset + (t + 1) * maxU + u] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, t, u, blank_
                )
                emit = betas[offset + t * maxU + u + 1] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, t, u, labels[u]
                )
                betas[offset + t * maxU + u] = rnnt_helper.log_sum_exp(emit, no_emit)

        # sync across all B=b and U=u
        cuda.syncthreads()

    # After final sync, betas[b, 0, 0] gives
    # log-likelihood of backward pass.
    if u == 0:
        llBackward[b] = betas[offset]


@cuda.jit()
def compute_grad_kernel(
    grads: torch.Tensor,
    acts: torch.Tensor,
    denom: torch.Tensor,
    alphas: torch.Tensor,
    betas: torch.Tensor,
    logll: torch.Tensor,
    xlen: torch.Tensor,
    ylen: torch.Tensor,
    mlabels: torch.Tensor,  # [B, U]
    minibatch: int,
    maxT: int,
    maxU: int,
    alphabet_size: int,
    blank_: int,
):
    """
    Compute gradients over the transduction step.

    Args:
        grads: Zero Tensor of shape [B, T, U, V+1]. Is updated by this kernel to contain the gradients
            of this batch of samples.
        acts: Tensor of shape [B, T, U, V+1] flattened. Represents the logprobs activation tensor.
        denom: Tensor of shape [B, T, U] flattened. Represents the denominator of the logprobs activation tensor
            across entire vocabulary.
        alphas: Alpha variable, contains forward probabilities. A tensor of shape [B, T, U].
        betas: Beta varoable, contains backward probabilities. A tensor of shape [B, T, U].
        logll: Log-likelihood of the forward variable, represented as a vector of shape [B].
            Represents the log-likelihood of the forward pass.
        xlen: Vector of length B which contains the actual acoustic sequence lengths in the padded
            activation tensor.
        ylen: Vector of length B which contains the actual target sequence lengths in the padded
            activation tensor.
        mlabels: Matrix of shape [B, U+1] (+1 here is due to <SOS> token - usually the RNNT blank).
            The matrix contains the padded target transcription that must be predicted.
        minibatch: Int representing the batch size.
        maxT: The maximum possible acoustic sequence length. Represents T in the logprobs tensor.
        maxU: The maximum possible target sequence length. Represents U in the logprobs tensor.
        alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT blank).
        blank_: Index of the RNNT blank token in the vocabulary. Generally the first or last token in the vocab.

    Updates:
        Kernel inplace updates the following inputs:
        -   grads: Gradients with respect to the log likelihood (logll).
    """
    # Kernel call:
    # blocks_per_grid = minibatch (b) * maxT (t) * maxU (u)
    # threads_per_block = constant buffer size of parallel threads (v :: Constant)
    tid = cuda.threadIdx.x  # represents v, taking steps of some constant size
    idx = tid  # index of v < V+1; in steps of constant buffer size
    col = cuda.blockIdx.x  # represents a fused index of b * t * u

    # Decompose original indices from fused `col`
    u = col % maxU  # (b * t * u) % u = u
    bt = (col - u) // maxU  # (b * t * u - u) // U = b * t
    t = bt % maxT  # (b * t) % t = t
    mb = (bt - t) // maxT  # (b * t - t) // T = b

    # constants
    T = xlen[mb]  # select AM length of current sample
    U = ylen[mb] + 1  # select target length of current sample, +1 for the blank token
    labels: torch.Tensor = mlabels[mb]  # labels = mlabels + mb * (maxU - 1);

    # Buffered gradient calculations, broadcast across B=b, T=t and U=u, looped over V with some constant stride.
    # Look up gradient calculation from rnnt_numpy.compute_gradient()
    if t < T and u < U:
        # For cuda kernels, maximum number of threads per block is limited to some value.
        # However, it may be the case that vocabulary size is larger than this limit
        # To work around this, an arbitrary thread buffer size is chosen such that,
        # 1) each element within the thread pool operates independently of the other
        # 2) An inner while loop moves the index of each buffer element by the size of the buffer itself,
        #    such that all elements of the vocabulary size are covered in (V + 1 // thread_buffer) number of steps.
        # As such, each thread will perform the while loop at least (V + 1 // thread_buffer) number of times
        while idx < alphabet_size:
            # remember, `col` represents the tri-index [b, t, u]
            # therefore; logpk = denom[b, t, u] + acts[b, t, u, v]
            logpk = denom[col] + acts[col * alphabet_size + idx]
            # initialize the grad of the sample acts[b, t, u, v]
            grad = math.exp(alphas[col] + betas[col] + logpk - logll[mb])

            # // grad to last blank transition
            # grad[b, T-1, U-1, v=blank] -= exp(alphas[b, t, u) + logpk - logll[b])
            if (idx == blank_) and (t == T - 1) and (u == U - 1):
                grad -= math.exp(alphas[col] + logpk - logll[mb])

            # grad of blank across t < T;
            # grad[b, t<T-1, u, v=blank] -= exp(alphas[b, t, u] + logpk - logll[b] betas[b, t + 1, u])
            if (idx == blank_) and (t < T - 1):
                grad -= math.exp(alphas[col] + logpk - logll[mb] + betas[col + maxU])

            # grad of correct token across u < U;
            # grad[b, t, u<U-1, v=label[u]] -= exp(alphas[b, t, u] + logpk - logll[b] + betas[b, t, u+1])
            if (u < U - 1) and (idx == labels[u]):
                grad -= math.exp(alphas[col] + logpk - logll[mb] + betas[col + 1])

            # update grads[b, t, u, v] = grad
            grads[col * alphabet_size + idx] = grad

            # update internal index through the thread_buffer;
            # until idx < V + 1, such that entire vocabulary has been updated.
            idx += GPU_RNNT_THREAD_SIZE
