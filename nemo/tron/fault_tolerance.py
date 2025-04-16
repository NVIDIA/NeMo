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

"""
Fault Tolerance (FT) package integration for Megatron-LM, using the FT section-based API.

The FT package is included in "nvidia-resiliency-ext"
(https://github.com/NVIDIA/nvidia-resiliency-ext).

NOTE: The workload must be run using the `ft_launcher` tool provided by `nvidia-resiliency-ext.`
NOTE: Calls to the public API of this module are no-ops if FT is not initialized
(`ft_integration.setup` was not called).
NOTE: Default distributed process group should be initialized before calling `ft_integration.setup`

The "setup" FT section is opened during FT initialization and closed before the first training or
eval iteration. Training and evaluation steps are wrapped in the "step" section, but only after a
few warmup iterations. This is because the initial iterations may be slower, and we want the "step"
timeout to be short. These warmup steps, which are not wrapped in the "step" section, will fall into
the out-of-section area. All checkpoint-saving-related operations (including asynchronous
checkpointing finalization) are wrapped in the "checkpointing" section.

If timeout calculation is enabled (--calc-ft-timeouts),
FT timeouts are updated after each checkpoint and at the end of the run.
Updated values are based on observed intervals.

`ft_launcher` command example:
```
ft_launcher \
    --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes=${NUM_NODES} --nproc-per-node=${NUM_GPUS_PER_NODE} \
    --ft-param-rank_section_timeouts=setup:600,step:180,checkpointing:420 \
    --ft-param-rank_out_of_section_timeout=300 \
    train_script_with_ft.py
```
"""

import json
import os
import random
import signal
import sys
import threading
import time
from typing import List, Optional

import torch

from nemo.tron.config import ConfigContainer, FaultToleranceConfig
from nemo.tron.state import GlobalState
from nemo.tron.utils.common_utils import get_rank_safe, print_rank_0

_NUM_WARMUP_ITERS = 1
_MIN_ITERS_FOR_STEP_TIMEOUT_UPDATE = 16


def setup(config: ConfigContainer, global_state: GlobalState) -> None:
    """Initialize fault tolerance integration.

    Opens the 'setup' FT section.

    Args:
        config: Configuration container.
        global_state: Global training state.

    Raises:
        ValueError: If checkpoint save directory is not configured.
    """
    from nvidia_resiliency_ext.fault_tolerance import RankMonitorClient

    print_rank_0("FT: initializing...")

    checkpoint_dir = config.checkpoint_config.save
    if not checkpoint_dir:
        raise ValueError("checkpointing save dir must be set to enable fault tolerance")
    if get_rank_safe() == 0 and not os.path.exists(checkpoint_dir):
        # MLM checkpoint dir will be needed for saving FT state.
        # it can happen before the checkpointing, so create it in advance
        os.makedirs(checkpoint_dir, exist_ok=True)

    global_state.rank_monitor_client = RankMonitorClient()

    global_state.fault_tolerance_state.ft_state_path = os.path.join(checkpoint_dir, "ft_state.json")

    global_state.fault_tolerance_state.is_async_chkpt_enabled = config.checkpoint_config.async_save

    global_state.fault_tolerance_state.is_calculating_timeouts = config.ft_config.calc_ft_timeouts

    global_state.rank_monitor_client.init_workload_monitoring()
    _load_state_if_exists(global_state)
    print_rank_0(f"FT: initialized. Timeouts={global_state.rank_monitor_client.section_timeouts}")

    global_state.rank_monitor_client.start_section("setup")
    global_state.fault_tolerance_state.is_setup_section_open = True


def on_training_step_start(global_state: GlobalState) -> None:
    """Callback executed at the start of each training step.

    Closes the 'setup' section if open, and starts the 'step' section
    after warmup iterations.

    Args:
        global_state: Global training state.
    """
    rmon_cli = global_state.rank_monitor_client
    ft_state = global_state.fault_tolerance_state
    if rmon_cli is not None:
        if ft_state.is_setup_section_open:
            rmon_cli.end_section("setup")
            ft_state.is_setup_section_open = False
        if ft_state.seen_tr_iters_cnt >= _NUM_WARMUP_ITERS:
            rmon_cli.start_section("step")
        # reset eval step index. we started training, so evaluation is done
        ft_state.curr_eval_iter_idx = 0


def on_training_step_end(global_state: GlobalState) -> None:
    """Callback executed at the end of each training step.

    Ends the 'step' section if it was started.

    Args:
        global_state: Global training state.
    """
    rmon_cli = global_state.rank_monitor_client
    ft_state = global_state.fault_tolerance_state
    if rmon_cli is not None:
        if ft_state.seen_tr_iters_cnt >= _NUM_WARMUP_ITERS:
            rmon_cli.end_section("step")
        ft_state.seen_tr_iters_cnt += 1


def on_eval_step_start(global_state: GlobalState) -> None:
    """Callback executed at the start of each evaluation step.

    Closes the 'setup' section if open, and starts the 'step' section
    after warmup iterations.

    Args:
        global_state: Global training state.
    """
    rmon_cli = global_state.rank_monitor_client
    ft_state = global_state.fault_tolerance_state
    if rmon_cli is not None:
        if ft_state.is_setup_section_open:
            # setup section can be open if there were no training iters before evaluation
            rmon_cli.end_section("setup")
            ft_state.is_setup_section_open = False
        if ft_state.curr_eval_iter_idx >= _NUM_WARMUP_ITERS:
            rmon_cli.start_section("step")


def on_eval_step_end(global_state: GlobalState) -> None:
    """Callback executed at the end of each evaluation step.

    Ends the 'step' section if it was started.

    Args:
        global_state: Global training state.
    """
    rmon_cli = global_state.rank_monitor_client
    ft_state = global_state.fault_tolerance_state
    if rmon_cli is not None:
        if ft_state.curr_eval_iter_idx >= _NUM_WARMUP_ITERS:
            rmon_cli.end_section("step")
        ft_state.curr_eval_iter_idx += 1


def on_checkpointing_start(global_state: GlobalState) -> None:
    """Callback executed before checkpoint-saving related operations.

    Starts the 'checkpointing' FT section.

    Args:
        global_state: Global training state.
    """
    rmon_cli = global_state.rank_monitor_client
    if rmon_cli is not None:
        rmon_cli.start_section("checkpointing")


def on_checkpointing_end(is_async_finalization: bool, global_state: GlobalState) -> None:
    """Callback executed after checkpoint-saving related operations.

    Ends the 'checkpointing' FT section and potentially updates timeouts.

    Args:
        is_async_finalization: True if called after async checkpoint finalization.
        global_state: Global training state.
    """
    rmon_cli = global_state.rank_monitor_client
    ft_state = global_state.fault_tolerance_state
    if rmon_cli is not None:
        rmon_cli.end_section("checkpointing")
    # async checkpointing finalization is called before each training iter, it can be no-op.
    # let's try to update the timeouts only on the `save_checkpoint`
    if not is_async_finalization:
        ft_state.seen_checkpoints_cnt += 1
        _maybe_update_timeouts(global_state)


def on_checkpoint_loaded(is_local_chkpt: bool, global_state: GlobalState) -> None:
    """Callback executed after a checkpoint is loaded.

    Records whether a persistent checkpoint was loaded for timeout calculation.

    Args:
        is_local_chkpt: True if a local (non-persistent) checkpoint was loaded.
        global_state: Global training state.
    """
    ft_state = global_state.fault_tolerance_state
    # checkpoint can be loaded during "setup"
    # check if persistent checkpoint was loaded,
    # in-memory checkpoint reading can be very fast,
    # so we could underestimate the "setup" timeout
    ft_state.is_persistent_chkpt_loaded = not is_local_chkpt


def shutdown(global_state: GlobalState) -> None:
    """Shuts down fault tolerance monitoring.

    Updates timeouts if applicable and closes the FT client.

    Args:
        global_state: Global training state.
    """
    rmon_cli = global_state.rank_monitor_client
    if rmon_cli is not None:
        print_rank_0("FT: closing...")
        _maybe_update_timeouts(global_state, is_closing_ft=True)
        rmon_cli.shutdown_workload_monitoring()
        print_rank_0("FT: closed.")
    global_state.rank_monitor_client = None


def maybe_setup_simulated_fault(config: FaultToleranceConfig) -> None:
    """Sets up a simulated fault for fault tolerance testing, if configured.

    Starts a background thread that will hang or kill a specific rank after a delay.

    Args:
        config: Fault tolerance configuration object.
    """
    if not config.simulate_fault:
        return

    if config.simulated_fault_type == "random":
        fault_type = random.choice(["rank_hung", "rank_killed"])
    else:
        fault_type = config.simulated_fault_type

    if config.simulated_fault_rank is None:
        fault_rank = random.randint(0, torch.distributed.get_world_size() - 1)
    else:
        fault_rank = config.simulated_fault_rank

    print_rank_0(f"Setting up simulated fault: type={fault_type}, rank={fault_rank}")

    def __fault_thread():
        # Add a small random delay to avoid all ranks failing at exactly the same time
        time.sleep(config.simulated_fault_base_delay + random.random())
        if get_rank_safe() == fault_rank:
            if fault_type == "rank_hung":
                print_rank_0(f"Simulating rank {fault_rank} hang by sleeping forever")
                while True:
                    time.sleep(1)
            elif fault_type == "rank_killed":
                print_rank_0(f"Simulating rank {fault_rank} killed by sending SIGKILL")
                os.kill(os.getpid(), signal.SIGKILL)

    threading.Thread(target=__fault_thread, daemon=True).start()


# Private functions below
def _load_state_if_exists(global_state: GlobalState) -> None:
    """Load fault tolerance state from file if it exists."""
    rmon_cli = global_state.rank_monitor_client
    ft_state = global_state.fault_tolerance_state
    if get_rank_safe() == 0 and os.path.exists(ft_state.ft_state_path):
        with open(ft_state.ft_state_path, "r") as f:
            state = json.load(f)
            rmon_cli.section_timeouts = state["section_timeouts"]
            rmon_cli.out_of_section_timeout = state["out_of_section_timeout"]


def _update_timeouts(selected_sections: List[str], calc_out_of_section: bool, global_state: GlobalState) -> None:
    """Update fault tolerance timeouts based on observed intervals."""
    rmon_cli = global_state.rank_monitor_client
    ft_state = global_state.fault_tolerance_state
    if get_rank_safe() == 0:
        state = {
            "section_timeouts": rmon_cli.section_timeouts,
            "out_of_section_timeout": rmon_cli.out_of_section_timeout,
        }
        with open(ft_state.ft_state_path, "w") as f:
            json.dump(state, f)


def _maybe_update_timeouts(global_state: GlobalState, is_closing_ft: bool = False) -> None:
    """Update timeouts if conditions are met."""
    ft_state = global_state.fault_tolerance_state
    if not ft_state.is_calculating_timeouts:
        return

    # we need to see enough iterations to estimate the step timeout
    if ft_state.seen_tr_iters_cnt < _MIN_ITERS_FOR_STEP_TIMEOUT_UPDATE:
        return

    # we need to see at least one checkpoint to estimate the checkpointing timeout
    if ft_state.seen_checkpoints_cnt == 0:
        return

    # we need to see at least one persistent checkpoint load to estimate the setup timeout
    if not ft_state.is_persistent_chkpt_loaded:
        return

    # we need to see at least one async checkpoint finalization to estimate the checkpointing timeout
    if ft_state.is_async_chkpt_enabled and not is_closing_ft:
        return

    selected_sections = ["setup", "step", "checkpointing"]
    calc_out_of_section = True
    _update_timeouts(selected_sections, calc_out_of_section, global_state)
