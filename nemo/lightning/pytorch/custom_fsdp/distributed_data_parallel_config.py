# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Optional


@dataclass
class DistributedDataParallelConfig:
    """Configuration for DistributedDataParallel."""

    grad_reduce_in_fp32: bool = False
    """If true, reduce grads in fp32."""

    overlap_grad_reduce: bool = False
    """If true, overlap grad all-reduce / reduce-scatter with backward compute."""

    overlap_param_gather: bool = False
    """If true, overlap param all-gather with forward compute."""

    align_param_gather: bool = False
    """If true, all PP stages will launch param all-gathers simultaneously. Otherwise, each
    PP stage will independently launch as needed.
    """

    use_distributed_optimizer: bool = False
    """If true, issue reduce-scatter collectives to aggregate gradients and clean up
       originally allocated model parameters, otherwise issue all-reduce collectives.
    """

    num_distributed_optimizer_instances: int = 1
    """Sets the factor by which the DP domain is sharded to have the partial DistOpt
       enabled. Defaults to 1, which means DistOpt is across entire DP domain.
    """

    check_for_nan_in_grad: bool = False
    """If true, check for NaNs and Infs in gradients _before_ communication collective."""

    check_for_large_grads: bool = False
    """If true, check for unexpectedly large gradients _before_ communication collective."""

    bucket_size: Optional[int] = None
    """Maximum number of parameters in each bucket. If unspecified, MCore uses a default
       value of max(40000000, 1000000 * dp_size) parameters (larger DP sizes need larger
       buckets to ensure collectives do not become latency-bound)."""

    pad_buckets_for_high_nccl_busbw: bool = False
    """If true, make sure the bucket size is divisible by a large power of 2 (2^16) to
       ensure NCCL collectives have high bus bandwidth at large DP counts, since NCCL
       message size (which for ring algorithms is bucket_size / dp_size) apparently needs
       to be divisible by a power of 2 for high busbw."""

    average_in_collective: bool = False
    """If true, compute average in collective directly, as opposed to dividing by the
       dp_size first and then computing sum in the collective."""

    fp8_param_gather: bool = False
    """If true, keep the compute param in fp8 (do not use any other intermediate dtype) and
       perform the param all-gather in fp8."""

    use_custom_fsdp: bool = False
    """If true, use the FSDP code path for DDP."""

    data_parallel_sharding_strategy: str = 'no_shard'
    """Sharding strategy for FSDP. Valid values are 'no_shard', 'optim',
        'optim_grads', 'optim_grads_params'."""

    gradient_reduce_div_fusion: bool = True
    """If true, perform gradient reduce and division fusion."""

    suggested_communication_unit_size: int = None
    """Specifies the number of elements to communicate at once during
      FSDP (Fully Sharded Data Parallel) operations. 
      This flag also affects FSDP all-gather prefetch behavior. Setting a larger
      value increases the communication buffer size, while a smaller value
      disables prefetching and may degrade performance. Adjust this value
      based on your system's memory and performance requirements."""

    preserve_fp32_weights: bool = True
    """If true, preserve fp32 weights in the custom FSDP ParamAndGradBuffer."""

    keep_fp8_transpose_cache_when_using_custom_fsdp: bool = False
    """If true, keep the fp8 transpose cache when using custom FSDP."""
