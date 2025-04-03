# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import collections
import collections.abc
import ctypes
import functools
import os
from datetime import timedelta
from typing import Any, Callable, Optional

import pynvml
import torch
import torch.distributed as dist

from cosmos1.utils import log
from cosmos1.utils.device import Device


def init() -> int | None:
    """Initialize distributed training."""
    # Set GPU affinity.
    pynvml.nvmlInit()
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = Device(local_rank)
    os.sched_setaffinity(0, device.get_cpu_affinity())
    # Set up NCCL communication.
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    if dist.is_available():
        if dist.is_initialized():
            return torch.cuda.current_device()
        torch.cuda.set_device(local_rank)
        # Get the timeout value from environment variable
        timeout_seconds = os.getenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", 1800)
        # Convert the timeout to an integer (if it isn't already) and then to a timedelta
        timeout_timedelta = timedelta(seconds=int(timeout_seconds))
        dist.init_process_group(backend="nccl", init_method="env://", timeout=timeout_timedelta)
        log.critical(
            f"Initialized distributed training with local rank {local_rank} with timeout {timeout_seconds}",
            rank0_only=False,
        )
    # Increase the L2 fetch granularity for faster speed.
    _libcudart = ctypes.CDLL("libcudart.so")
    # Set device limit on the current device.
    p_value = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(p_value, ctypes.c_int(0x05))
    log.info(f"Training with {get_world_size()} GPUs.")


def get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    """Get the rank (GPU device) of the worker.

    Returns:
        rank (int): The rank of the worker.
    """
    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank(group)
    return rank


def get_world_size(group: Optional[dist.ProcessGroup] = None) -> int:
    """Get world size. How many GPUs are available in this job.

    Returns:
        world_size (int): The total number of GPUs available in this job.
    """
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size(group)
    return world_size


def is_rank0() -> bool:
    """Check if current process is the master GPU.

    Returns:
        (bool): True if this function is called from the master GPU, else False.
    """
    return get_rank() == 0


def rank0_only(func: Callable) -> Callable:
    """Apply this function only to the master GPU.

    Example usage:
        @rank0_only
        def func(x):
            return x + 3

    Args:
        func (Callable): a function.

    Returns:
        (Callable): A function wrapper executing the function only on the master GPU.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN202
        if is_rank0():
            return func(*args, **kwargs)
        else:
            return None

    return wrapper


def barrier() -> None:
    """Barrier for all GPUs."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    """This extends torch.nn.parallel.DistributedDataParallel with .training_step().

    This borrows the concept of `forward-redirection` from Pytorch lightning. It wraps an coreModel such that
    model.training_step() would be executed when calling self.training_step(), while preserving the behavior of calling
    model() for Pytorch modules. Internally, this is a double rerouting mechanism (training_step -> forward ->
    training_step), allowing us to preserve the function names and signatures.
    """

    def __init__(self, model: torch.nn.Module, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def training_step(self, *args, **kwargs) -> Any:
        # Cache the original model.forward() method.
        original_forward = self.module.forward

        def wrapped_training_step(*_args, **_kwargs):  # noqa: ANN202
            # Unpatch immediately before calling training_step() because itself may want to call the real forward.
            self.module.forward = original_forward
            # The actual .training_step().
            return self.module.training_step(*_args, **_kwargs)

        # Patch the original_module's forward so we can redirect the arguments back to the real method.
        self.module.forward = wrapped_training_step
        # Call self, which implicitly calls self.forward() --> model.forward(), which is now model.training_step().
        # Without calling self.forward() or model.forward() explciitly, implicit hooks are also executed.
        return self(*args, **kwargs)


def collate_batches(data_batches: list[dict[str, torch.Tensor]]) -> torch.Tensor | dict[str, torch.Tensor]:
    """Aggregate the list of data batches from all devices and process the results.

    This is used for gathering validation data batches with utils.dataloader.DistributedEvalSampler.
    It will return the data/output of the entire validation set in its original index order. The sizes of data_batches
    in different ranks may differ by 1 (if dataset size is not evenly divisible), in which case a dummy sample will be
    created before calling dis.all_gather().

    Args:
        data_batches (list[dict[str, torch.Tensor]]): List of tensors or (hierarchical) dictionary where
            leaf entries are tensors.

    Returns:
        data_gather (torch.Tensor | dict[str, torch.Tensor]): tensors or (hierarchical) dictionary where
            leaf entries are concatenated tensors.
    """
    if isinstance(data_batches[0], torch.Tensor):
        # Concatenate the local data batches.
        data_concat = torch.cat(data_batches, dim=0)  # type: ignore
        # Get the largest number of local samples from all ranks to determine whether to dummy-pad on this rank.
        max_num_local_samples = torch.tensor(len(data_concat), device="cuda")
        dist.all_reduce(max_num_local_samples, op=dist.ReduceOp.MAX)
        if len(data_concat) < max_num_local_samples:
            assert len(data_concat) + 1 == max_num_local_samples
            dummy = torch.empty_like(data_concat[:1])
            data_concat = torch.cat([data_concat, dummy], dim=0)
            dummy_count = torch.tensor(1, device="cuda")
        else:
            dummy_count = torch.tensor(0, device="cuda")
        # Get all concatenated batches from all ranks and concatenate again.
        dist.all_reduce(dummy_count, op=dist.ReduceOp.SUM)
        data_concat = all_gather_tensor(data_concat.contiguous())
        data_collate = torch.stack(data_concat, dim=1).flatten(start_dim=0, end_dim=1)
        # Remove the dummy samples.
        if dummy_count > 0:
            data_collate = data_collate[:-dummy_count]
    elif isinstance(data_batches[0], collections.abc.Mapping):
        data_collate = dict()
        for key in data_batches[0].keys():
            data_collate[key] = collate_batches([data[key] for data in data_batches])  # type: ignore
    else:
        raise TypeError
    return data_collate


@torch.no_grad()
def all_gather_tensor(tensor: torch.Tensor) -> list[torch.Tensor]:
    """Gather the corresponding tensor from all GPU devices to a list.

    Args:
        tensor (torch.Tensor): Pytorch tensor.

    Returns:
        tensor_list (list[torch.Tensor]): A list of Pytorch tensors gathered from all GPU devices.
    """
    tensor_list = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(tensor_list, tensor)
    return tensor_list


def broadcast(tensor, src, group=None, async_op=False):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    dist.broadcast(tensor, src=src, group=group, async_op=async_op)
