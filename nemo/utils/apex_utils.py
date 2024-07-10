import warnings
from typing import List, Optional

import torch
from megatron.core.num_microbatches_calculator import build_num_microbatches_calculator


def _reconfigure_microbatch_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
) -> None:
    if torch.distributed.get_rank() == 0:
        import warnings

        warnings.warn("This function is only for unittest")

    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR

    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(
        rank, rampup_batch_size, global_batch_size, micro_batch_size, data_parallel_size
    )


def get_micro_batch_size():
    from megatron.core.num_microbatches_calculator import _GLOBAL_NUM_MICROBATCHES_CALCULATOR

    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.micro_batch_size
