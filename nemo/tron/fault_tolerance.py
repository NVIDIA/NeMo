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

import argparse
import json
import os
import random
import signal
import sys
import threading
import time
from typing import Any, Optional

import torch

from . import global_vars
from .utils import is_rank0, print_rank_0

_GLOBAL_RANK_MONITOR_CLIENT = None

_ft_state_path = None
_is_persistent_chkpt_loaded = False
_is_async_chkpt_enabled = False
_is_calculating_timeouts = False
_is_setup_section_open = False
_seen_checkpoints_cnt = 0
_seen_tr_iters_cnt = 0
_curr_eval_iter_idx = 0

_NUM_WARMUP_ITERS = 1
_MIN_ITERS_FOR_STEP_TIMEOUT_UPDATE = 16


def get_rank_monitor_client() -> Optional[Any]:
    """Returns the underlying fault tolerance client instance

    Returns:
        RankMonitorClient: rank monitor client instance, or None if FT was not initialized
    """
    return _GLOBAL_RANK_MONITOR_CLIENT


def setup(args: argparse.Namespace) -> None:
    """Initialize fault tolerance

    Args:
        args (argparse.Namespace): parsed Megatron-LM command line arguments

    Raises:
        ValueError: if invalid config is provided
    """
    from nvidia_resiliency_ext.fault_tolerance import RankMonitorClient

    print_rank_0(f"FT: initializing...")

    checkpoint_dir = args.save
    if not checkpoint_dir:
        raise ValueError("checkpointing save dir must be set to enable fault tolerance")
    if is_rank0() and not os.path.exists(checkpoint_dir):
        # MLM checkpoint dir will be needed for saving FT state.
        # it can happen before the checkpointing, so create it in advance
        os.makedirs(checkpoint_dir, exist_ok=True)

    cli = RankMonitorClient()
    global _GLOBAL_RANK_MONITOR_CLIENT
    global_vars._ensure_var_is_not_initialized(_GLOBAL_RANK_MONITOR_CLIENT, "rank monitor client")
    _GLOBAL_RANK_MONITOR_CLIENT = cli

    global _ft_state_path
    _ft_state_path = os.path.join(checkpoint_dir, "ft_state.json")

    global _is_async_chkpt_enabled
    _is_async_chkpt_enabled = args.async_save

    global _is_calculating_timeouts
    _is_calculating_timeouts = args.calc_ft_timeouts

    cli.init_workload_monitoring()
    _load_state_if_exists()
    print_rank_0(f"FT: initialized. Timeouts={cli.section_timeouts}")

    cli.start_section("setup")
    global _is_setup_section_open
    _is_setup_section_open = True


def on_training_step_start() -> None:
    """Should be called before each training step"""
    rmon_cli = get_rank_monitor_client()
    if rmon_cli is not None:
        global _is_setup_section_open
        if _is_setup_section_open:
            rmon_cli.end_section("setup")
            _is_setup_section_open = False
        if _seen_tr_iters_cnt >= _NUM_WARMUP_ITERS:
            rmon_cli.start_section("step")
        # reset eval step index. we started training, so evaluation is done
        global _curr_eval_iter_idx
        _curr_eval_iter_idx = 0


def on_training_step_end() -> None:
    """Should be called after each training step"""
    rmon_cli = get_rank_monitor_client()
    if rmon_cli is not None:
        global _seen_tr_iters_cnt
        if _seen_tr_iters_cnt >= _NUM_WARMUP_ITERS:
            rmon_cli.end_section("step")
        _seen_tr_iters_cnt += 1


def on_eval_step_start() -> None:
    """Should be called before each validation step"""
    rmon_cli = get_rank_monitor_client()
    if rmon_cli is not None:
        global _is_setup_section_open
        if _is_setup_section_open:
            # setup section can be open if there were no training iters before evaluation
            rmon_cli.end_section("setup")
            _is_setup_section_open = False
        if _curr_eval_iter_idx >= _NUM_WARMUP_ITERS:
            rmon_cli.start_section("step")


def on_eval_step_end() -> None:
    """Should be called after each validation step"""
    rmon_cli = get_rank_monitor_client()
    if rmon_cli is not None:
        global _curr_eval_iter_idx
        if _curr_eval_iter_idx >= _NUM_WARMUP_ITERS:
            rmon_cli.end_section("step")
        _curr_eval_iter_idx += 1


def on_checkpointing_start() -> None:
    """Should be called before each checkpoint-saving-related operation."""
    rmon_cli = get_rank_monitor_client()
    if rmon_cli is not None:
        rmon_cli.start_section("checkpointing")


def on_checkpointing_end(is_async_finalization: bool) -> None:
    """Should be called after each checkpoint-saving-related operation.

    Args:
        is_async_finalization (bool): true if called after an async checkpointing finalization
    """
    rmon_cli = get_rank_monitor_client()
    if rmon_cli is not None:
        rmon_cli.end_section("checkpointing")
    # async checkpointing finalization is called before each training iter, it can be no-op.
    # let's try to update the timeouts only on the `save_checkpoint`
    if not is_async_finalization:
        global _seen_checkpoints_cnt
        _seen_checkpoints_cnt += 1
        _maybe_update_timeouts()


def on_checkpoint_loaded(is_local_chkpt: bool) -> None:
    """Should be called after a checkpoint was loaded

    Args:
        is_local_chkpt (bool): true if it was a local checkpoint, false if global
    """
    # checkpoint can be loaded during "setup"
    # check if persistent checkpoint was loaded,
    # in-memory checkpoint reading can be very fast,
    # so we could underestimate the "setup" timeout
    global _is_persistent_chkpt_loaded
    _is_persistent_chkpt_loaded = not is_local_chkpt


def shutdown() -> None:
    """Shutdowns fault folerance, updates the FT timeouts if possible"""
    global _GLOBAL_RANK_MONITOR_CLIENT
    rmon_cli = get_rank_monitor_client()
    if rmon_cli is not None:
        print_rank_0("FT: closing...")
        _maybe_update_timeouts(is_closing_ft=True)
        rmon_cli.shutdown_workload_monitoring()
        print_rank_0("FT: closed.")
    _GLOBAL_RANK_MONITOR_CLIENT = None


def _load_state_if_exists():
    rmon_cli = get_rank_monitor_client()
    if os.path.exists(_ft_state_path):
        with open(_ft_state_path, "r") as f:
            ft_state = json.load(f)
        rmon_cli.load_state_dict(ft_state)
        print_rank_0(f"FT: loaded timeouts from {_ft_state_path}. {rmon_cli.section_timeouts}")


def _update_timeouts(selected_sections, calc_out_of_section):
    print_rank_0(
        f"FT: updating timeouts for: {selected_sections} " + f"update out-of-section: {calc_out_of_section} ..."
    )
    rmon_cli = get_rank_monitor_client()
    rmon_cli.calculate_and_set_section_timeouts(
        selected_sections=selected_sections, calc_out_of_section=calc_out_of_section
    )
    if is_rank0():
        ft_state = rmon_cli.state_dict()
        with open(_ft_state_path, "w") as f:
            json.dump(ft_state, f)
        print_rank_0(f"FT: updated timeouts saved to {_ft_state_path}. {rmon_cli.section_timeouts}")


def _maybe_update_timeouts(is_closing_ft=False):
    rmon_cli = get_rank_monitor_client()
    if rmon_cli is None:
        return
    if not _is_calculating_timeouts:
        return

    # Decide which section timeouts can be updated
    sections_to_update = []

    if _is_persistent_chkpt_loaded:
        sections_to_update.append("setup")
    else:
        print_rank_0("FT: can't update the setup section timeout until persistent checkpoint is loaded")

    if _seen_tr_iters_cnt >= _MIN_ITERS_FOR_STEP_TIMEOUT_UPDATE:
        sections_to_update.append("step")
    else:
        print_rank_0("FT: need to see more training iterations to update the step section timeout")

    if _seen_checkpoints_cnt > 0:
        if not _is_async_chkpt_enabled:
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
        if {"setup", "step"}.issubset(sections_to_update) and _seen_checkpoints_cnt > 0:
            update_out_of_section = True
        else:
            print_rank_0("FT: the out-of-section timeout won't be updated until all FT sections were seen")

    else:
        print_rank_0("FT: the out-of-section timeout won't be updated as the FT is not closing yet")

    if sections_to_update or update_out_of_section:
        _update_timeouts(selected_sections=sections_to_update, calc_out_of_section=update_out_of_section)


def maybe_setup_simulated_fault() -> None:
    """Sets a simulated fault, based on `FT_SIM_FAULT_DESC` env variable.
    Simulated fault description format:
    rank_hung|rank_killed;rank_to_fail|"";base_delay
    NOTE: This if for FT testing only
    """

    simulated_fault_desc = os.environ.get("FT_SIM_FAULT_DESC", None)
    if not simulated_fault_desc:
        return
    fault_type: Any  # silence mypy
    rank_to_fail: Any  # silence mypy
    base_delay: Any  # silence mypy
    fault_type, rank_to_fail, base_delay = simulated_fault_desc.split(";")
    fault_type = fault_type.strip()
    rank_to_fail = rank_to_fail.strip()
    rank_to_fail = int(rank_to_fail) if rank_to_fail else None
    base_delay = float(base_delay.strip())

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
