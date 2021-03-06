import math
import torch
import numba
from numba import cuda

from nemo.collections.asr.parts.numba.rnnt_loss.utils import rnnt_helper
from nemo.collections.asr.parts.numba.rnnt_loss.utils import global_constants


GPU_RNNT_THREAD_SIZE = 128


@cuda.jit(device=True, inline=True)
def logp(
    denom: torch.Tensor, acts: torch.Tensor, maxT: int, maxU: int, alphabet_size: int, mb: int, t: int, u: int, v: int
):
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
    # // launch B blocks, each block has U threads
    b = cuda.blockIdx.x  # // batch id
    u = cuda.threadIdx.x  # label id, u
    T = xlen[b]
    U = ylen[b] + 1  # +1 for the blank token

    labels: torch.Tensor = mlabels[b]  # equivalent to mlabels + b * (maxU - 1); // mb label start point
    offset = b * maxT * maxU

    # alphas = alphas[offset:]  # alphas += offset # pointer offset

    if u == 0:
        alphas[offset] = 0

    cuda.syncthreads()
    for n in range(1, T + U - 1):
        t = n - u
        if u == 0:
            if t > 0 and t < T:
                alphas[offset + t * maxU + u] = alphas[offset + (t - 1) * maxU + u] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, t - 1, 0, blank_
                )
        elif u < U:
            if t == 0:
                alphas[offset + u] = alphas[offset + u - 1] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, 0, u - 1, labels[u - 1]
                )
            elif t > 0 and t < T:
                no_emit = alphas[offset + (t - 1) * maxU + u] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, t - 1, u, blank_
                )
                emit = alphas[offset + t * maxU + u - 1] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, t, u - 1, labels[u - 1]
                )

                alphas[offset + t * maxU + u] = rnnt_helper.log_sum_exp(emit, no_emit)

        cuda.syncthreads()

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
    # // launch B blocks, each block has U threads
    b = cuda.blockIdx.x  # // batch id
    u = cuda.threadIdx.x  # label id, u
    T = xlen[b]
    U = ylen[b] + 1  # +1 for the blank token

    labels: torch.Tensor = mlabels[b]  # equivalent to mlabels + b * (maxU - 1); // mb label start point
    offset = b * maxT * maxU

    # betas = betas[offset:]  # betas += offset # pointer offset

    if u == 0:
        betas[offset + (T - 1) * maxU + U - 1] = logp(denom, acts, maxT, maxU, alphabet_size, b, T - 1, U - 1, blank_)

    cuda.syncthreads()
    for n in range(T + U - 2, -1, -1):
        t = n - u
        if u == (U - 1):
            if t >= 0 and t < (T - 1):
                betas[offset + t * maxU + U - 1] = betas[offset + (t + 1) * maxU + U - 1] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, t, U - 1, blank_
                )
        elif u < U:
            if t == T - 1:
                betas[offset + (T - 1) * maxU + u] = betas[offset + (T - 1) * maxU + u + 1] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, T - 1, u, labels[u]
                )
            elif (t >= 0) and (t < T - 1):
                no_emit = betas[offset + (t + 1) * maxU + u] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, t, u, blank_
                )
                emit = betas[offset + t * maxU + u + 1] + logp(
                    denom, acts, maxT, maxU, alphabet_size, b, t, u, labels[u]
                )
                betas[offset + t * maxU + u] = rnnt_helper.log_sum_exp(emit, no_emit)

        cuda.syncthreads()

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
    tid = cuda.threadIdx.x
    idx = tid
    col = cuda.blockIdx.x

    u = col % maxU
    bt = (col - u) // maxU
    t = bt % maxT
    mb = (bt - t) // maxT

    # constants
    T = xlen[mb]
    U = ylen[mb] + 1
    labels: torch.Tensor = mlabels[mb]  # labels = mlabels + mb * (maxU - 1);

    if t < T and u < U:
        while idx < alphabet_size:
            logpk = denom[col] + acts[col * alphabet_size + idx]
            grad = math.exp(alphas[col] + betas[col] + logpk - logll[mb])

            # // grad to last blank transition
            if (idx == blank_) and (t == T - 1) and (u == U - 1):
                grad -= math.exp(alphas[col] + logpk - logll[mb])

            if (idx == blank_) and (t < T - 1):
                grad -= math.exp(alphas[col] + logpk - logll[mb] + betas[col + maxU])

            if (u < U - 1) and (idx == labels[u]):
                grad -= math.exp(alphas[col] + logpk - logll[mb] + betas[col + 1])

            grads[col * alphabet_size + idx] = grad
            idx += GPU_RNNT_THREAD_SIZE
