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

import inspect
import os
from datetime import datetime
from typing import Any, Optional

import torch.distributed
import yaml
from megatron.core import DistributedDataParallel as DDP
from megatron.core.transformer.module import Float16Module

from nemo.tron.utils.yaml_utils import safe_yaml_representers

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, torch_FSDP, Float16Module)
except ImportError:
    ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


def unwrap_model(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def use_dist_ckpt(ckpt_format: str) -> bool:
    return ckpt_format != "torch"


def get_rank_safe() -> int:
    # In megatron init, args.rank comes from the torchrun env var.
    # Once init has been done, args.rank is updated to value of torch get_rank()
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return int(os.getenv("RANK", "0"))


def get_world_size_safe() -> int:
    # In megatron init, args.world_size comes from the torchrun env var.
    # Once init has been done, args.world_size is updated to value of torch get_world_size()
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return int(os.getenv("WORLD_SIZE", "1"))


def get_local_rank_preinit() -> int:
    return int(os.getenv("LOCAL_RANK", "0"))


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    rank = get_rank_safe()
    if rank == 0:
        print(message, flush=True)


def is_last_rank():
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)


def append_to_progress_log(save_dir: str, string: str, barrier: bool = True):
    """Append given string to progress log."""
    if save_dir is None:
        return
    progress_log_filename = os.path.join(save_dir, "progress.txt")
    if barrier and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if get_rank_safe() == 0:
        os.makedirs(os.path.dirname(progress_log_filename), exist_ok=True)
        with open(progress_log_filename, "a+") as f:
            job_id = os.getenv("SLURM_JOB_ID", "")
            num_gpus = get_world_size_safe()
            f.write(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\tJob ID: {job_id}\t# GPUs: {num_gpus}\t{string}\n"
            )


def barrier_and_log(string):
    """Note that this call will sync across all ranks."""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_rank_0(f"[{string}] datetime: {time_str} ")


def _safe_object_representer(dumper, data):
    """
    Represent a given object as YAML using the specified dumper.

    This function is a fallback for objects that don't have specific representers.
    If the object has __qualname__ attr,
    the __target__ is set to f"{inspect.getmodule(obj).__name__}.{obj.__qualname__}".
    If the object does not have a __qualname__ attr, the __target__ is set from its __class__ attr.
    The __call__ key is used to indicate whether the target should be called to create an instance.

    Args:
        dumper (yaml.Dumper): The YAML dumper to use for serialization.
        data (Any): The data to serialize. This can be any Python object,
            but if it's a class or a class instance, special handling will be applied.

    Returns:
        str: The YAML representation of the data.
    """
    try:
        obj = data
        target = f"{inspect.getmodule(obj).__name__}.{obj.__qualname__}"
        call = False
    except AttributeError:
        obj = data.__class__
        target = f"{inspect.getmodule(obj).__name__}.{obj.__qualname__}"
        call = True

    value = {
        "_target_": target,  # type: ignore
        "_call_": call,
    }
    return dumper.represent_data(value)


def dump_dataclass_to_yaml(obj: Any, filename: Optional[str] = None):
    with safe_yaml_representers():
        if filename is not None:
            with open(filename, "w+") as f:
                yaml.safe_dump(obj, f)
        else:
            return yaml.safe_dump(obj)
