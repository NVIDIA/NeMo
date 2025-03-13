# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import socket
import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Literal, Optional

import torch
import torch.distributed as dist
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.parallel_state import destroy_global_memory_buffer, destroy_model_parallel
from megatron.core.rerun_state_machine import destroy_rerun_state_machine

from nemo.utils.import_utils import safe_import

inprocess, HAVE_RES = safe_import('nvidia_resiliency_ext.inprocess')


@dataclass
class InProcessRestartConfig:
    monitor_thread_interval: float = 1.0
    monitor_process_interval: float = 1.0
    progress_watchdog_interval: float = 1.0
    heartbeat_interval: float = 30.0
    soft_timeout: float = 60.0
    hard_timeout: float = 120.0
    heartbeat_timeout: float = 60.0
    barrier_timeout: float = 180.0
    completion_timeout: float = 180.0
    last_call_wait: float = 1.0
    termination_grace_time: float = 5.0
    cuda_health_check_timeout: float = 10.0
    monitor_process_logfile: Optional[str] = None
    enabled: bool = True

    empty_cuda_cache: bool = True
    active_world_size: int = int(os.getenv("WORLD_SIZE", "1"))
    granularity: Literal["node", "rank"] = "node"


def get_tcp_store() -> dist.Store:
    """Create TCPStore for distributed communication if using wrapper."""

    # Uses (MASTER_PORT + 1) to avoid conflicts with torch.distributed.run
    return dist.TCPStore(
        host_name=os.environ.get('MASTER_ADDR', 'localhost'),
        port=int(os.environ.get('MASTER_PORT', 29500)) + 1,
        world_size=int(os.environ.get('WORLD_SIZE', 1)),
        is_master=int(os.environ.get('RANK', 0)) == 0,
        multi_tenant=True,
        wait_for_workers=True,
        use_libuv=True,
    )


def get_prefix_store(store: dist.Store, call_wrapper: inprocess.CallWrapper) -> dist.Store:
    iteration = call_wrapper.iteration
    return dist.PrefixStore(str(iteration), store)


def get_finalize_fns(config: InProcessRestartConfig) -> List[inprocess.finalize.ThreadedFinalize]:
    finalize_fns = [
        inprocess.finalize.ThreadedFinalize(
            timeout=timedelta(seconds=10),
            fn=_destroy_mcore_global_state,
        )
    ]
    if config.empty_cuda_cache:
        finalize_fns.append(
            inprocess.finalize.ThreadedFinalize(
                timeout=timedelta(seconds=10),
                fn=torch.cuda.empty_cache,
            )
        )
    return finalize_fns


def _destroy_mcore_global_state() -> None:
    destroy_num_microbatches_calculator()
    destroy_global_memory_buffer()
    destroy_model_parallel()
    destroy_rerun_state_machine()


def get_rank_assignment_layers(config: InProcessRestartConfig) -> List[inprocess.rank_assignment.Layer]:
    layers = [
        inprocess.rank_assignment.Layer(
            min_ranks=config.active_world_size,
            max_ranks=config.active_world_size,
            flag=inprocess.rank_assignment.LayerFlag.RESERVE,
        )
    ]
    if config.granularity == "node":
        device_count = torch.cuda.device_count()
        layers.append(
            inprocess.rank_assignment.Layer(
                min_ranks=device_count,
                max_ranks=device_count,
                key_or_fn=lambda _: socket.gethostname(),
                flag=inprocess.rank_assignment.LayerFlag.RESERVE,
            )
        )
    return layers


def maybe_set_torch_cpp_log_level_error() -> None:
    if "TORCH_CPP_LOG_LEVEL" not in os.environ or os.environ['TORCH_CPP_LOG_LEVEL'] not in ("error", "fatal"):
        warnings.warn('Setting TORCH_CPP_LOG_LEVEL=error to suppress c10d waitForInput timeout warning messages')
        os.environ["TORCH_CPP_LOG_LEVEL"] = "error"
