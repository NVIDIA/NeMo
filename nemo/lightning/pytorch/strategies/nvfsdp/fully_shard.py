# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import types
from enum import IntEnum
from typing import Optional, Sequence, Type, TypeVar

import torch
from torch.distributed import DeviceMesh

from .nvfsdp import nvFSDP
from .utils import FSDPDistributedIndex, create_updated_function_signature


try:
    # Default to Megatron-LM FW.
    from megatron.core.distributed.distributed_data_parallel_config import (
        DistributedDataParallelConfig,
    )
except ImportError:
    # Megatron-LM is not installed, use nvFSDP as a standalone module.
    from .distributed_data_parallel_config import DistributedDataParallelConfig


class ShardingStrategy(IntEnum):
    """
    IntEnum to track the abbreviated sharding strategy for nvFSDP.

    - `0` or `no_shard` implies that your model is not sharded. Similar memory usage to `DDP`.
    - `1` or `optim` implies that your optimizer state is sharded. Similar to optimizer state sharding in `ZeRO-DP`.
    - `2` or `optim_grads` implies that your optimizer state and gradients are sharded. Similar to optimizer state and gradient sharding in `ZeRO-2`.
    - `3` or `optim_grads_params` implies that your optimizer state, gradients, and training parameters are sharded. Similar to optimizer state, gradient, and training parameter sharding in `ZeRO-3`.
    """

    NO_SHARD = 0
    OPTIM = 1
    OPTIM_GRADS = 2
    OPTIM_GRADS_PARAMS = 3


# Hints input-output consistency - if an optimizer is provided, it must be returned.
# If an optimizer is not provided, None will be returned.
MaybeOptimizer = TypeVar("MaybeOptimizer", bound=Optional[torch.optim.Optimizer])


def fully_shard(
    module: torch.nn.Module,
    optimizer: MaybeOptimizer = None,
    data_parallel_sharding_strategy: str | int = 3,
    fsdp_unit_modules: Optional[Sequence[Type[torch.nn.Module]] | Sequence[str]] = None,
    device_mesh: Optional[DeviceMesh] = None,
    dp_mesh_dim_name: Optional[str] = None,
    cp_mesh_dim_name: Optional[str] = None,
    tp_mesh_dim_name: Optional[str] = None,
    dp_cp_mesh_dim_name: Optional[str] = None,
    expt_dp_group: Optional[torch.distributed.ProcessGroup] = None,
    device: Optional[torch.device] = None,
    init_model_with_meta_device: bool = True,
    grad_reduce_in_fp32: bool = False,
    preserve_fp32_weights: bool = True,
    overlap_grad_reduce: bool = True,
    overlap_param_gather: bool = True,
    sync_grads_each_step: bool = True,
    check_for_nan_in_grad: bool = True,
    average_in_collective: bool = False,
    disable_bucketing: bool = False,
    calculate_per_token_loss: bool = False,
    keep_fp8_transpose_cache_when_using_custom_fsdp: bool = False,
    nccl_ub: bool = False,
    fsdp_double_buffer: bool = False,
) -> tuple[nvFSDP, MaybeOptimizer]:
    """
    Fully shard the model and the optimizer for nvFSDP.

    Wraps the model as an nvFSDP module, and modifies the optimizer to
    be compatible with the nvFSDP training strategy.

    Args:
        module (torch.nn.Module):
            The PyTorch module fully-sharded and managed by nvFSDP.

        optimizer (Optional[torch.optim.Optimizer]):
            (Distributed) optimizer for training the model, which is extended to automatically
            execute necessary nvFSDP operations during the training loop. If not provided, the
            user is expected to utilize the nvFSDP API to manually prepare the model for
            optimization. Defaults to None.

        data_parallel_sharding_strategy (str | int):
            Strategy for sharding data parallel parameters and gradients. Options include:
            - "no_shard" / 0: No optimizer, gradient, or parameter sharding. Similar memory usage to DDP.
            - "optim" / 1: Shards optimizer states (and main weights for mixed precision training),
                which is conceptually similar to optimizer state sharding in `ZeRO-DP`.
            - "optim_grads" / 2: Shards gradients and optimizer states, which is conceptually
                similar to "ZeRO-2".
            - "optim_grads_params" / 3: Shards parameters, gradients and optimizer states, which
                is conceptually similar to "ZeRO-3".
            Defaults to "optim_grads_params" / 3.

        fsdp_unit_modules (Optional[List[torch.nn.Module] | List[str]]):
            List of module classes or module class import paths to be treated as FSDP units,
            which are modules that do not have their parameters modified during their forward().
            This information is utilized by nvFSDP to shard, gather, and overlap communications
            during the forward and backward pass of the module. Defaults to None.

        device_mesh (Optional[DeviceMesh]):
            Device mesh object defining the topology for distributed training. Defaults to None,
            in which case the {dp,cp,tp,expt}_group(s) are required to use nvFSDP. If device_mesh is None,
            nvFSDP will automatically use torch.distributed.group.WORLD for sharded data parallelism.

        dp_mesh_dim_name (Optional[str]):
            Name of the data parallel sub-mesh in the device_mesh. Used to identify the
            data parallel dimension for sharding. Defaults to None.

        cp_mesh_dim_name (Optional[str]):
            Name of the context parallel sub-mesh in the device_mesh. Used to identify the
            context parallel dimension for sharding. Defaults to None.

        tp_mesh_dim_name (Optional[str]):
            Name of the tensor parallel sub-mesh in the device_mesh. Used to identify the
            tensor parallel dimension for sharding. Defaults to None.

        dp_cp_mesh_dim_name (Optional[str]):
            Name of the DP-CP sub-mesh in the device_mesh. Used to identify the flattened
            data parallel / context parallel dimension for sharding. Defaults to None.
            Users that want to optimize the NCCL configuration for this process group
            can provide their custom DP-CP sub-mesh using this argument.

        expt_dp_group (Optional[torch.distributed.ProcessGroup]):
            Megatron Core Expert Parallelism process group. Used to identify the expert parallel
            dimension for sharding expert modules only. Not compatible with DTensor-based TP.
            Defaults to None.

        device (Optional[torch.device]):
            Target device for the sharded model. Used to migrate all parameters in the model
            to an expected device. If init_model_with_meta_device=True, this argument is ignored.
            Defaults to None.

        init_model_with_meta_device (bool):
            Whether to initialize model parameters in shards across all devices in the fsdp_group.
            Utilized to initialize large models that do not fit on a single device.
            Defaults to True.

        grad_reduce_in_fp32 (bool):
            Whether to perform gradient reduction in FP32. Defaults to False.

        preserve_fp32_weights (bool):
            Whether to preserve FP32 optimization weights. Defaults to True.

        overlap_grad_reduce (bool):
            Whether to overlap gradient reduce-scatter (or all-reduce) with backward compute.
            Defaults to True.

        overlap_param_gather (bool):
            Whether to overlap parameter all-gather with forward and backward compute.
            Defaults to True.

        sync_grads_each_step (bool):
            Whether to synchronize and install optimizer gradients on each training step.
            When disabled, nvFSDP will overlap reduce-scatter calls with subsequent compute,
            which improves performance and throughput when utilizing delayed optimization
            techniques such as gradient accumulation. Defaults to True for the fully_shard API.

        check_for_nan_in_grad (bool):
            Whether to check for NaN values in gradients. Defaults to True.

        average_in_collective (bool):
            Whether to average gradients in collective communication. Defaults to False.
            TODO: This is currently NOT supported!

        disable_bucketing (bool):
            Whether to disable gradient bucketing optimization, which permits more granular
            and precise communication of parameters and gradients. Defaults to False.

        calculate_per_token_loss (bool):
            Whether to calculate loss per token, which deactivates gradient scaling.
            Defaults to False.

        keep_fp8_transpose_cache_when_using_custom_fsdp (bool):
            Whether to keep the FP8 transpose cache when using a custom FSDP.
            Defaults to False.

        nccl_ub (bool):
            Whether to use NCCL UCC for communication. Defaults to False.

        fsdp_double_buffer (bool):
            Whether to use double buffer for FSDP. Defaults to False.

    Returns:
        torch.nn.Module: The wrapped nvFSDP model configured for distributed training.
        torch.optim.Optimizer: The nvFSDP-compliant optimizer for training the model.

    Note:
        This implementation uses NVIDIA's FSDP which includes optimizations specific
        to NVIDIA hardware and software stack.
    """

    # Parse data_parallel_sharding_strategy.
    if data_parallel_sharding_strategy == ShardingStrategy.NO_SHARD:
        data_parallel_sharding_strategy = "no_shard"
    elif data_parallel_sharding_strategy == ShardingStrategy.OPTIM:
        data_parallel_sharding_strategy = "optim"
    elif data_parallel_sharding_strategy == ShardingStrategy.OPTIM_GRADS:
        data_parallel_sharding_strategy = "optim_grads"
    elif data_parallel_sharding_strategy == ShardingStrategy.OPTIM_GRADS_PARAMS:
        data_parallel_sharding_strategy = "optim_grads_params"

    if data_parallel_sharding_strategy not in [
        "no_shard",
        "optim",
        "optim_grads",
        "optim_grads_params",
    ]:
        raise ValueError(
            f"Invalid nvFSDP Sharding Strategy: {data_parallel_sharding_strategy}\n"
            f"Valid Sharding Strategies: {ShardingStrategy.NO_SHARD}, {ShardingStrategy.OPTIM}, {ShardingStrategy.OPTIM_GRADS}, {ShardingStrategy.OPTIM_GRADS_PARAMS}, "
            f"no_shard, optim, optim_grads, optim_grads_params"
        )

    # DDP Config for Custom FSDP
    ddp_config = DistributedDataParallelConfig(
        check_for_nan_in_grad=check_for_nan_in_grad,
        data_parallel_sharding_strategy=data_parallel_sharding_strategy,
        grad_reduce_in_fp32=grad_reduce_in_fp32,
        preserve_fp32_weights=preserve_fp32_weights,
        overlap_grad_reduce=overlap_grad_reduce,
        overlap_param_gather=overlap_param_gather,
        average_in_collective=average_in_collective,
        keep_fp8_transpose_cache_when_using_custom_fsdp=keep_fp8_transpose_cache_when_using_custom_fsdp,
        nccl_ub=nccl_ub,
        fsdp_double_buffer=fsdp_double_buffer,
    )

    # Create FSDPDistributedIndex.
    dist_index = FSDPDistributedIndex(
        device_mesh=device_mesh,
        dp_mesh_dim_name=dp_mesh_dim_name,
        cp_mesh_dim_name=cp_mesh_dim_name,
        tp_mesh_dim_name=tp_mesh_dim_name,
        dp_cp_mesh_dim_name=dp_cp_mesh_dim_name,
        expt_dp_group=expt_dp_group,
    )

    # Wrap model in Custom FSDP.
    model = nvFSDP(
        module=module,
        dist_index=dist_index,
        ddp_config=ddp_config,
        fsdp_unit_modules=fsdp_unit_modules,
        disable_bucketing=disable_bucketing,
        device=device,
        calculate_per_token_loss=calculate_per_token_loss,
        init_model_with_meta_device=init_model_with_meta_device,
        sync_grads_each_step=sync_grads_each_step,
    )

    # Extend optimizer methods to support nvFSDP operations.
    if optimizer is not None:
        # Replace the optimizer module parameter references with the nvFSDP-managed parameters.
        optimizer.param_groups.clear()
        optimizer.state.clear()
        optimizer.add_param_group({"params": model.module.parameters()})

        # Save a reference to the optimizer.step() and optimizer.zero_grad() methods.
        optimizer_step_base_func = type(optimizer).step
        optimizer_zero_grad_base_func = type(optimizer).zero_grad

        # Define a new optimizer.step() method that distributes optimizer state and gradients,
        # waits for asynchronous gradient reduce-scatter work to be completed, and updates model weights.
        def nvfsdp_optimizer_step(optimizer, *args, **kwargs):
            # Extract extended kwargs.
            sync_grad_before_optimizer_step = kwargs.pop(
                # If sync_grads_each_step is enabled, gradients are synchronized automatically
                # during the post-backward hook, so we do not need to synchronize them here.
                "sync_grad_before_optimizer_step",
                not sync_grads_each_step,
            )
            install_optimized_model_weights = kwargs.pop("install_optimized_model_weights", True)

            # Synchronize reduce-scatter and all-gather operations for all model gradients
            # and parameters, attach gradients to the optimizer state, and replace the raw
            # module parameters with nvFSDP-managed optimizer parameters & states in
            # preparation for (distributed) optimization.
            if sync_grad_before_optimizer_step:
                model.finish_grad_sync()

            # Execute the base optimizer.step() on the model optimizer named parameters.
            optimizer_step_base_func(optimizer, *args, **kwargs)

            # Update the raw module training parameters with optimized values.
            if install_optimized_model_weights:
                model.install_optimized_model_weights()

        # Define a new optimizer.zero_grad() method that zeros the gradient in both
        # the optimizer as well as the nvFSDP gradient buffer.
        def nvfsdp_optimizer_zero_grad(optimizer, *args, **kwargs):
            # Extract extended kwargs.
            zero_grad_buffer = kwargs.pop("zero_grad_buffer", True)

            # Execute the base optimizer.zero_grad() on the model optimizer named parameters.
            optimizer_zero_grad_base_func(optimizer, *args, **kwargs)

            # Zero out the gradient in the nvFSDP gradient buffer.
            if zero_grad_buffer:
                model.zero_grad_buffer()

        # Override the optimizer.step() and optimizer.zero_grad() methods to support nvFSDP operations.
        nvfsdp_optimizer_step.__signature__ = create_updated_function_signature(
            optimizer_step_base_func,
            sync_grad_before_optimizer_step=True,
            install_optimized_model_weights=True,
        )
        optimizer.step = types.MethodType(nvfsdp_optimizer_step, optimizer)
        nvfsdp_optimizer_zero_grad.__signature__ = create_updated_function_signature(
            optimizer_zero_grad_base_func,
            zero_grad_buffer=True,
        )
        optimizer.zero_grad = types.MethodType(nvfsdp_optimizer_zero_grad, optimizer)

    # Return model and optimizer.
    return model, optimizer
