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

import atexit
import datetime
import signal

import torch
import torch.distributed
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator, init_num_microbatches_calculator)

from nemo.automodel.config import DistributedInitConfig
from nemo.tron.config import TrainingConfig
from nemo.tron.utils.common_utils import (get_local_rank_preinit,
                                          get_rank_safe, get_world_size_safe)


def initialize_automodel(
    dist_config: DistributedInitConfig,
    training_config: TrainingConfig,
    data_parallel_size: int,
    seed: int,
    allow_no_cuda: bool = False,
    skip_dist_initialization: bool = False,
):
    """Initialize megatron global vars, logging, and distributed state."""

    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), "Megatron requires CUDA."

    # TODO: Add rampup_batch_size support if needed
    init_num_microbatches_calculator(  # noqa: F821
        get_rank_safe(),
        rampup_batch_size=None,
        global_batch_size=training_config.global_batch_size,
        micro_batch_size=training_config.micro_batch_size,
        data_parallel_size=data_parallel_size,
    )

    # torch.distributed initialization
    return torch_dist_init(
        dist_config=dist_config,
        seed=seed,
        skip_dist_initialization=skip_dist_initialization,
    )


def torch_dist_init(
    dist_config: DistributedInitConfig,
    seed: int,
    skip_dist_initialization: bool,
):
    def finish_dist_init():
        # Pytorch distributed.
        _initialize_distributed(dist_config=dist_config)

        # Random seeds for reproducibility.
        if get_rank_safe() == 0:
            print("> setting random seeds to {} ...".format(seed))
        _set_random_seed(seed=seed)

    if skip_dist_initialization:
        return None

    if dist_config.lazy_init:
        return finish_dist_init
    else:
        # Complete initialization right away.
        finish_dist_init()
        # No continuation function
        return None


def _initialize_distributed(
    dist_config: DistributedInitConfig,
):
    """Initialize torch.distributed and core model parallel."""

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        if get_rank_safe() == 0:
            print(
                "torch distributed is already initialized, skipping initialization ...",
                flush=True,
            )

    else:
        if get_rank_safe() == 0:
            print("> initializing torch distributed ...", flush=True)

        # Manually set the device ids.
        if device_count > 0:
            torch.cuda.set_device(get_local_rank_preinit())

        # Call the init process
        init_process_group_kwargs = {
            "backend": dist_config.distributed_backend,
            "world_size": get_world_size_safe(),
            "rank": get_rank_safe(),
            "timeout": datetime.timedelta(minutes=dist_config.distributed_timeout_minutes),
        }

        if get_world_size_safe() == 1:
            import socket

            def find_free_port():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", 0))
                    return s.getsockname()[1]

            free_port = find_free_port()
            init_process_group_kwargs["world_size"] = 1
            init_process_group_kwargs["rank"] = 0
            init_process_group_kwargs["init_method"] = f"tcp://localhost:{free_port}"

        torch.distributed.init_process_group(**init_process_group_kwargs)
        atexit.register(destroy_global_state)
        torch.distributed.barrier(device_ids=[get_local_rank_preinit()])


def _set_random_seed(seed: int):
    """Set random seed for reproducability."""
    assert seed is not None and seed > 0, f"Seed ({seed}) should be a positive integer."

    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # TODO: add cuda seed


def destroy_global_state():
    destroy_num_microbatches_calculator()
    # Don't allow Ctrl+C to interrupt this handler
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
