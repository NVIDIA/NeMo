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

import os
from datetime import datetime

import torch.distributed


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


def append_to_progress_log(save_dir: str, string: str, barrier: bool = True):
    """Append given string to progress log."""
    if save_dir is None:
        return
    progress_log_filename = os.path.join(save_dir, "progress.txt")
    if barrier:
        torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        with open(progress_log_filename, "a") as f:
            job_id = os.getenv("SLURM_JOB_ID", "")
            num_gpus = get_world_size_safe()
            f.write(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\tJob ID: {job_id}\t# GPUs: {num_gpus}\t{string}\n"
            )


def barrier_and_log(string):
    """Note that this call will sync across all ranks."""
    torch.distributed.barrier()
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_rank_0(f"[{string}] datetime: {time_str} ")
