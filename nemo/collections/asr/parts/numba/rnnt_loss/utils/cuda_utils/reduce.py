import math
import enum
import torch
from numba import cuda

from nemo.collections.asr.parts.numba.rnnt_loss.utils import rnnt_helper
from nemo.collections.asr.parts.numba.rnnt_loss.utils import global_constants

warp_size = global_constants.warp_size()
dtype = global_constants.dtype()

CTA_REDUCE_SIZE = 128


class I_Op(enum.Enum):
    EXPONENTIAL = 0
    IDENTITY = 1


class R_Op(enum.Enum):
    ADD = 0
    MAXIMUM = 1


@cuda.jit(device=True)
def CTAReduce(tid: int, x, storage, count: int, R_opid: int):
    storage[tid] = x

    cuda.syncthreads()

    # Fold the data in half with each pass
    offset = CTA_REDUCE_SIZE // 2
    while offset >= warp_size:
        if (tid + offset) < count and tid < offset:
            # Read from the right half and store to the left half.
            if R_opid == 0:
                x = rnnt_helper.add(x, storage[offset + tid])
            else:
                x = rnnt_helper.maximum(x, storage[offset + tid])

            storage[tid] = x

        cuda.syncthreads()
        offset = offset // 2

    offset = warp_size // 2
    while offset > 0:
        # warp sync
        shuff = cuda.shfl_down_sync(0xFFFFFFFF, x, offset)

        if (tid + offset < count) and (tid < offset):
            if R_opid == 0:
                x = rnnt_helper.add(x, shuff)
            else:
                x = rnnt_helper.maximum(x, shuff)

        offset = offset // 2

    return x


@cuda.jit()
def _reduce_rows(I_opid: int, R_opid: int, acts, output, num_rows: int):
    tid = cuda.threadIdx.x
    idx = tid
    col = cuda.blockIdx.x

    storage = cuda.shared.array(shape=(CTA_REDUCE_SIZE,), dtype=acts.dtype)

    max = output[col]

    # // Each block works on a column
    if idx < num_rows:
        curr = acts[col * num_rows + idx] - max
        if I_opid == 0:
            curr = rnnt_helper.exponential(curr)
        else:
            curr = rnnt_helper.identity(curr)

    idx += CTA_REDUCE_SIZE

    while idx < num_rows:
        activation_ = acts[col * num_rows + idx] - max

        if I_opid == 0 and R_opid == 0:
            curr = rnnt_helper.add(curr, rnnt_helper.exponential(activation_))
        elif I_opid == 0 and R_opid == 1:
            curr = rnnt_helper.maximum(curr, rnnt_helper.exponential(activation_))
        elif I_opid == 1 and R_opid == 0:
            curr = rnnt_helper.add(curr, rnnt_helper.identity(activation_))
        else:
            curr = rnnt_helper.maximum(curr, rnnt_helper.identity(activation_))

        idx += CTA_REDUCE_SIZE

    # // Sum thread-totals over the CTA.
    curr = CTAReduce(tid, curr, storage, num_rows, R_opid)

    # // Store result in out
    if tid == 0:
        output[col] = curr


@cuda.jit()
def _reduce_minus(I_opid: int, R_opid: int, acts, output, num_rows: int):
    tid = cuda.threadIdx.x
    idx = tid
    col = cuda.blockIdx.x

    storage = cuda.shared.array(shape=(CTA_REDUCE_SIZE,), dtype=acts.dtype)

    max = output[col]

    # // Each block works on a column
    if idx < num_rows:
        curr = acts[col * num_rows + idx] - max
        if I_opid == 0:
            curr = rnnt_helper.exponential(curr)
        else:
            curr = rnnt_helper.identity(curr)

    idx += CTA_REDUCE_SIZE

    while idx < num_rows:
        activation_ = acts[col * num_rows + idx] - max

        if I_opid == 0 and R_opid == 0:
            curr = rnnt_helper.add(curr, rnnt_helper.exponential(activation_))
        elif I_opid == 0 and R_opid == 1:
            curr = rnnt_helper.maximum(curr, rnnt_helper.exponential(activation_))
        elif I_opid == 1 and R_opid == 0:
            curr = rnnt_helper.add(curr, rnnt_helper.identity(activation_))
        else:
            curr = rnnt_helper.maximum(curr, rnnt_helper.identity(activation_))

        idx += CTA_REDUCE_SIZE

    # // Sum thread-totals over the CTA.
    curr = CTAReduce(tid, curr, storage, num_rows, R_opid)

    # // Store result in out
    if tid == 0:
        output[col] = -max - math.log(curr)


def ReduceHelper(
    I_opid: int,
    R_opid: int,
    acts: torch.Tensor,
    output: torch.Tensor,
    num_rows: int,
    num_cols: int,
    minus: bool,
    stream,
):
    if minus:
        grid_size = num_cols
        # call kernel
        _reduce_minus[grid_size, CTA_REDUCE_SIZE, stream, 0](I_opid, R_opid, acts, output, num_rows)

    else:
        grid_size = num_cols
        # call kernel
        _reduce_rows[grid_size, CTA_REDUCE_SIZE, stream, 0](I_opid, R_opid, acts, output, num_rows)

    return True


def reduce_exp(acts: torch.Tensor, denom, rows: int, cols: int, minus: bool, stream):
    return ReduceHelper(
        I_opid=I_Op.EXPONENTIAL.value,
        R_opid=R_Op.ADD.value,
        acts=acts,
        output=denom,
        num_rows=rows,
        num_cols=cols,
        minus=minus,
        stream=stream,
    )


def reduce_max(acts: torch.Tensor, denom, rows: int, cols: int, minus: bool, stream):
    return ReduceHelper(
        I_opid=I_Op.IDENTITY.value,
        R_opid=R_Op.MAXIMUM.value,
        acts=acts,
        output=denom,
        num_rows=rows,
        num_cols=cols,
        minus=minus,
        stream=stream,
    )
