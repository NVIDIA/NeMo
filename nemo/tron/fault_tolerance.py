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

import torch

from nemo.tron.config import ConfigContainer, FaultToleranceConfig
from nemo.tron.state import GlobalState
from nemo.tron.utils.common_utils import get_rank_safe, print_rank_0

_NUM_WARMUP_ITERS = 1
_MIN_ITERS_FOR_STEP_TIMEOUT_UPDATE = 16


def setup(config: ConfigContainer, global_state: GlobalState) -> None:
    """Initialize fault tolerance

    Args:
        args (argparse.Namespace): parsed Megatron-LM command line arguments

    Raises:
        ValueError: if invalid config is provided
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
    """Should be called before each training step"""
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
    """Should be called after each training step"""
    rmon_cli = global_state.rank_monitor_client
    ft_state = global_state.fault_tolerance_state
    if rmon_cli is not None:
        if ft_state.seen_tr_iters_cnt >= _NUM_WARMUP_ITERS:
            rmon_cli.end_section("step")
        ft_state.seen_tr_iters_cnt += 1


def on_eval_step_start(global_state: GlobalState) -> None:
    """Should be called before each validation step"""
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
    """Should be called after each validation step"""
    rmon_cli = global_state.rank_monitor_client
    ft_state = global_state.fault_tolerance_state
    if rmon_cli is not None:
        if ft_state.curr_eval_iter_idx >= _NUM_WARMUP_ITERS:
            rmon_cli.end_section("step")
        ft_state.curr_eval_iter_idx += 1


def on_checkpointing_start(global_state: GlobalState) -> None:
    """Should be called before each checkpoint-saving-related operation."""
    rmon_cli = global_state.rank_monitor_client
    if rmon_cli is not None:
        rmon_cli.start_section("checkpointing")


def on_checkpointing_end(is_async_finalization: bool, global_state: GlobalState) -> None:
    """Should be called after each checkpoint-saving-related operation.

    Args:
        is_async_finalization (bool): true if called after an async checkpointing finalization
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
    """Should be called after a checkpoint was loaded

    Args:
        is_local_chkpt (bool): true if it was a local checkpoint, false if global
    """
    ft_state = global_state.fault_tolerance_state
    # checkpoint can be loaded during "setup"
    # check if persistent checkpoint was loaded,
    # in-memory checkpoint reading can be very fast,
    # so we could underestimate the "setup" timeout
    ft_state.is_persistent_chkpt_loaded = not is_local_chkpt


def shutdown(global_state: GlobalState) -> None:
    """Shutdowns fault folerance, updates the FT timeouts if possible"""
    rmon_cli = global_state.rank_monitor_client
    if rmon_cli is not None:
        print_rank_0("FT: closing...")
        _maybe_update_timeouts(global_state, is_closing_ft=True)
        rmon_cli.shutdown_workload_monitoring()
        print_rank_0("FT: closed.")
    global_state.rank_monitor_client = None


def _load_state_if_exists(global_state: GlobalState):
    rmon_cli = global_state.rank_monitor_client
    ft_state = global_state.fault_tolerance_state
    if os.path.exists(ft_state.ft_state_path):
        with open(ft_state.ft_state_path, "r") as f:
            rmon_state = json.load(f)
        rmon_cli.load_state_dict(rmon_state)
        print_rank_0(f"FT: loaded timeouts from {ft_state.ft_state_path}. {rmon_cli.section_timeouts}")


def _update_timeouts(selected_sections, calc_out_of_section, global_state: GlobalState):
    print_rank_0(
        f"FT: updating timeouts for: {selected_sections} " + f"update out-of-section: {calc_out_of_section} ..."
    )
    rmon_cli = global_state.rank_monitor_client
    ft_state = global_state.fault_tolerance_state
    rmon_cli.calculate_and_set_section_timeouts(
        selected_sections=selected_sections, calc_out_of_section=calc_out_of_section
    )
    if get_rank_safe() == 0:
        rmon_state = rmon_cli.state_dict()
        with open(ft_state.ft_state_path, "w") as f:
            json.dump(rmon_state, f)
        print_rank_0(f"FT: updated timeouts saved to {ft_state.ft_state_path}. {rmon_cli.section_timeouts}")


def _maybe_update_timeouts(global_state: GlobalState, is_closing_ft=False):
    rmon_cli = global_state.rank_monitor_client
    ft_state = global_state.fault_tolerance_state
    if rmon_cli is None:
        return
    if not ft_state.is_calculating_timeouts:
        return

    # Decide which section timeouts can be updated
    sections_to_update = []

    if ft_state.is_persistent_chkpt_loaded:
        sections_to_update.append("setup")
    else:
        print_rank_0("FT: can't update the setup section timeout until persistent checkpoint is loaded")

    if ft_state.seen_tr_iters_cnt >= _MIN_ITERS_FOR_STEP_TIMEOUT_UPDATE:
        sections_to_update.append("step")
    else:
        print_rank_0("FT: need to see more training iterations to update the step section timeout")

    if ft_state.seen_checkpoints_cnt > 0:
        if not ft_state.is_async_chkpt_enabled:
            sections_to_update.append("checkpointing")
        else:
            # There can be too much checkpointing section time variability
            # across runs with the async checkpointing, e.g. in some runs all checkpointing
            # work can be parallelized (=short checkpointing sections) while in others we can
            # hit a costly finalization.
            print_rank_0("FT: can't update the checkpointing section timeout with async checkpointing")
    else:
        print_rank_0("FT: checkpointing section is not updated until a checkpoint was saved")

    update_out_of_section = False
    if is_closing_ft:
        # with async checkpointing, "checkpointing" section is not updated,
        # but still we want to see some checkpointing to ensure that is was a complete run
        if {"setup", "step"}.issubset(sections_to_update) and ft_state.seen_checkpoints_cnt > 0:
            update_out_of_section = True
        else:
            print_rank_0("FT: the out-of-section timeout won't be updated until all FT sections were seen")

    else:
        print_rank_0("FT: the out-of-section timeout won't be updated as the FT is not closing yet")

    if sections_to_update or update_out_of_section:
        _update_timeouts(
            selected_sections=sections_to_update,
            calc_out_of_section=update_out_of_section,
            global_state=global_state,
        )


def maybe_setup_simulated_fault(config: FaultToleranceConfig) -> None:
    """Sets a simulated fault, based on `FT_SIM_FAULT_DESC` env variable.
    Simulated fault description format:
    rank_hung|rank_killed;rank_to_fail|"";base_delay
    NOTE: This if for FT testing only
    """

    if not config.simulate_fault:
        return
    fault_type = config.simulated_fault_type
    rank_to_fail = config.simulated_fault_rank
    base_delay = config.simulated_fault_base_delay

    rng = random.Random()

    print_rank_0(
        f"FT: Initializing simulated fault: {fault_type}," + f"rank to fail: {rank_to_fail}, base delay: {base_delay}"
    )

    # rank that simulates a fault can be explicitly specified in the `rank_to_fail` field
    # if not specified, it just picks a random rank
    rank = torch.distributed.get_rank()
    rand_rank = rng.randint(0, torch.distributed.get_world_size() - 1)
    rank_to_fail = rank_to_fail if rank_to_fail is not None else rand_rank
    rank_to_fail = torch.tensor([rank_to_fail], device=torch.cuda.current_device())
    torch.distributed.broadcast(rank_to_fail, 0)
    rank_to_fail = int(rank_to_fail.item())

    if rank != rank_to_fail:
        # this rank is not going to simulate a fault, nothing more to do
        return

    if fault_type == "random":
        fault_type = rng.choice(["rank_killed", "rank_hung"])

    if fault_type == "rank_killed":
        target_pid = os.getpid()
    elif fault_type == "rank_hung":
        target_pid = os.getpid()
    else:
        raise Exception(f"Unknown fault type {fault_type} expected one of: rank_killed, rank_hung.")

    # add some randomness to the delay
    delay = base_delay + 0.2 * rng.random() * base_delay

    print_rank_0(f"FT: Selected fault={fault_type}; target rank={rank_to_fail}; delay={delay}")

    def __fault_thread():
        time.sleep(delay)
        for of in [sys.stdout, sys.stderr]:
            print(
                f"\n####\nFT: Simulating fault: {fault_type}; rank to fail: {rank_to_fail}\n####\n",
                file=of,
                flush=True,
            )
        if fault_type == "rank_hung":
            os.kill(target_pid, signal.SIGSTOP)
        else:
            os.kill(target_pid, signal.SIGKILL)

    fault_sim_thread = threading.Thread(target=__fault_thread)
    fault_sim_thread.daemon = True
    fault_sim_thread.start()
