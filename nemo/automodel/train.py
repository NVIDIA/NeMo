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

import gc
import sys
import time
from datetime import datetime
from typing import Iterator

import torch
from megatron.core.num_microbatches_calculator import (
    get_current_global_batch_size,
    get_current_running_global_batch_size,
    get_num_microbatches,
    update_num_microbatches,
)

from nemo.automodel.checkpointing import save_checkpoint
from nemo.automodel.config import ConfigContainer
from nemo.automodel.eval import evaluate_and_print_results
from nemo.automodel.init import destroy_global_state
from nemo.automodel.schedules import ForwardStepFnProtocol, get_forward_backward_func
from nemo.automodel.utils import flop_utils
from nemo.automodel.utils.train_utils import reduce_loss, training_log
from nemo.tron.state import GlobalState
from nemo.tron.train import get_start_time_from_progress_log
from nemo.tron.utils.common_utils import append_to_progress_log, barrier_and_log, get_world_size_safe


def train_step(
    forward_step_func: ForwardStepFnProtocol,
    data_iterator: Iterator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    global_state: GlobalState,
):
    """Single training step."""
    cfg: ConfigContainer = global_state.cfg
    timers = global_state.timers
    train_config = cfg.train_config
    optim_config = cfg.optimizer_config

    optimizer.zero_grad()
    if hasattr(model, "zero_grad"):
        model.zero_grad()

    # Forward pass.
    forward_backward_func = get_forward_backward_func()
    loss_store, total_num_tokens = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        config=cfg.model_config,
        num_microbatches=get_num_microbatches(),
        forward_only=False,
        timers=timers,
    )
    # TODO: Add pipeline parallelism
    reporting_loss, total_num_tokens = reduce_loss(loss_store, total_num_tokens)

    # Empty unused memory.
    if train_config.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=optim_config.clip_grad)

    # Update parameters.
    timers("optimizer", log_level=1).start(barrier=optim_config.barrier_with_L1_time)
    optimizer.step()
    timers("optimizer").stop()

    # Update learning rate.
    increment = get_num_microbatches() * train_config.micro_batch_size * cfg.data_parallel_size
    scheduler.step(increment=increment)
    skipped_iter = 0

    # TODO: Add code to count zero gradients if needed

    # Empty unused memory.
    if train_config.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    return (
        {"lm loss": (reporting_loss, total_num_tokens)},
        skipped_iter,
        grad_norm,
    )


def post_training_step_callbacks(
    iteration,
    prof,
    config: ConfigContainer,
):
    """Run all post-training-step functions (e.g., FT heartbeats, GC)."""
    train_config = config.train_config

    # Bring CPU and GPU back in sync if on right iteration.
    if train_config.train_sync_interval and iteration % train_config.train_sync_interval == 0:
        torch.cuda.synchronize()

    # Profiling.
    if (
        config.profiling_config
        and config.profiling_config.profile
        and iteration == config.profiling_config.profile_step_end
        and torch.distributed.get_rank() in config.profiling_config.profile_ranks
    ):
        if config.profiling_config.use_pytorch_profiler:
            assert prof is not None
            prof.stop()
        else:
            torch.cuda.cudart().cudaProfilerStop()

    # Manual garbage collection.
    if train_config.manual_gc:
        if train_config.manual_gc_interval != 0 and iteration % train_config.manual_gc_interval == 0:
            gc.collect()


def compute_throughputs_and_append_to_progress_log(state: GlobalState):
    if state.cfg.checkpoint_config.save is None:
        return

    num_floating_point_operations_so_far = state.train_state.floating_point_operations_so_far

    # Compute job throughput.
    # completed at the start of job.
    job_throughput = (num_floating_point_operations_so_far - state.train_state.floating_point_operations_so_far) / (
        (time.time() - state.start_time) * 10**12 * get_world_size_safe()
    )

    # Compute cumulative throughput since jobs of this world size were launched.
    # `get_start_time_from_progress_log` returns start time and number of floating-point
    # operations of first job of this world size.
    start_time, start_num_floating_point_operations = get_start_time_from_progress_log(state.cfg)
    elapsed_time = (datetime.now() - start_time).total_seconds()
    cumulative_throughput = (num_floating_point_operations_so_far - start_num_floating_point_operations) / (
        elapsed_time * 10**12 * get_world_size_safe()
    )

    tokens_so_far = state.train_state.consumed_train_samples * state.cfg.dataset_config.seq_length
    saved_ckpt_prefix = "Saving async checkpoint" if state.cfg.checkpoint_config.async_save else "Saved checkpoint"
    append_to_progress_log(
        state.cfg.checkpoint_config.save,
        f"{saved_ckpt_prefix}\tIteration: {state.train_state.step}\t"
        f"Job throughput: {job_throughput:.1f} TFLOP/s/GPU\t"
        f"Cumulative throughput: {cumulative_throughput:.1f} TFLOP/s/GPU\t"
        f"Floating-point operations: {num_floating_point_operations_so_far:.2e}\t"
        f"Tokens (in billions): {tokens_so_far / 10**9:.2f}",
    )


def train(
    forward_step_func: ForwardStepFnProtocol,
    model,
    optimizer,
    scheduler,
    train_data_iterator,
    valid_data_iterator,
    global_state: GlobalState,
):
    config: ConfigContainer = global_state.cfg
    train_config = config.train_config
    timers = global_state.timers

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss_dict = {}

    num_floating_point_operations_since_last_log_event = 0.0

    timers("interval-time", log_level=0).start(barrier=True)
    report_memory_flag = True
    should_exit = False
    exit_code = 0

    if train_config.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert (
            train_config.manual_gc_interval >= 0
        ), "Manual garbage collection interval should be larger than or equal to 0"
        gc.disable()
        gc.collect()

    num_microbatches = get_num_microbatches()
    eval_duration = 0.0
    eval_iterations = 0

    prof = None
    prof_config = config.profiling_config
    if (
        prof_config
        and prof_config.profile
        and torch.distributed.get_rank() in prof_config.profile_ranks
        and prof_config.use_pytorch_profiler
    ):
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=max(prof_config.profile_step_start - 1, 0),
                warmup=1 if prof_config.profile_step_start > 0 else 0,
                active=prof_config.profile_step_end - prof_config.profile_step_start,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(config.logger_config.tensorboard_dir),
            record_shapes=True,
            with_stack=True,
        )
        prof.start()

    # Run training iterations till done.
    while global_state.train_state.step < train_config.train_iters:
        if prof_config:
            if prof_config.profile and torch.distributed.get_rank() in prof_config.profile_ranks:
                if prof_config.use_pytorch_profiler:
                    prof.step()
                elif global_state.train_state.step == prof_config.profile_step_start:
                    torch.cuda.cudart().cudaProfilerStart()
                    torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        # fault_tolerance.on_checkpointing_start(global_state)
        # maybe_finalize_async_save(ckpt_cfg=config.checkpoint_config, blocking=False)
        # fault_tolerance.on_checkpointing_end(global_state=global_state, is_async_finalization=True)

        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        update_num_microbatches(global_state.train_state.consumed_train_samples, consistency_check=False, verbose=True)
        if get_num_microbatches() != num_microbatches and global_state.train_state.step != 0:
            assert get_num_microbatches() > num_microbatches, (
                f"Number of microbatches should be increasing due to batch size rampup; "
                f"instead going from {num_microbatches} to {get_num_microbatches()}"
            )
            if config.checkpoint_config.save is not None:
                save_checkpoint_and_time(
                    global_state,
                    model,
                    optimizer,
                    scheduler,
                )
        num_microbatches = get_num_microbatches()
        update_num_microbatches(global_state.train_state.consumed_train_samples, consistency_check=True, verbose=True)

        # TODO: implement dummy train_step to fast forward train_data_iterator.
        # Completely skip iteration if needed.
        # if global_state.train_state.step in config.checkpoint_config.iterations_to_skip:
        #     # Dummy train_step to fast forward train_data_iterator.
        #     dummy_train_step(train_data_iterator)
        #     global_state.train_state.step += 1
        #     batch_size = parallel_state.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
        #     global_state.train_state.consumed_train_samples += batch_size
        #     global_state.train_state.skipped_train_samples += batch_size
        #     continue

        # Run training step.
        # fault_tolerance.on_training_step_start(global_state)
        loss_dict, skipped_iter, grad_norm = train_step(
            forward_step_func, train_data_iterator, model, optimizer, scheduler, global_state
        )
        # fault_tolerance.on_training_step_end(global_state)

        global_state.train_state.step += 1
        batch_size = config.data_parallel_size * train_config.micro_batch_size * get_num_microbatches()
        global_state.train_state.consumed_train_samples += batch_size
        num_skipped_samples_in_batch = get_current_global_batch_size() - get_current_running_global_batch_size()
        if train_config.decrease_batch_size_if_needed:
            assert num_skipped_samples_in_batch >= 0
        else:
            assert num_skipped_samples_in_batch == 0
        global_state.train_state.skipped_train_samples += num_skipped_samples_in_batch
        num_floating_point_operations_in_batch = flop_utils.num_floating_point_operations(
            config, config.model_config.hf_config, batch_size
        )
        global_state.train_state.floating_point_operations_so_far += num_floating_point_operations_in_batch
        num_floating_point_operations_since_last_log_event += num_floating_point_operations_in_batch

        # Logging.
        # TODO: Add support for loss scaling and mixed precision training.
        loss_scale = 1.0
        params_norm = None

        if config.logger_config.log_params_norm:
            params_norm = torch.nn.utils.get_total_norm(model.parameters())
        learning_rate = None
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]

        report_memory_flag = training_log(
            loss_dict=loss_dict,
            total_loss_dict=total_loss_dict,
            learning_rate=learning_rate,
            loss_scale=loss_scale,
            report_memory_flag=report_memory_flag,
            skipped_iter=skipped_iter,
            grad_norm=grad_norm,
            params_norm=params_norm,
            config=config,
            global_state=global_state,
        )

        if (
            global_state.train_state.do_valid
            and train_config.eval_interval
            and global_state.train_state.step % train_config.eval_interval == 0
        ):
            timers("interval-time").stop()
            if train_config.manual_gc and train_config.manual_gc_eval:
                # Collect all objects.
                gc.collect()
            prefix = f"iteration {global_state.train_state.step}"
            timers("eval-time", log_level=0).start(barrier=True)
            evaluate_and_print_results(
                global_state,
                prefix,
                forward_step_func,
                valid_data_iterator,
                model,
                verbose=False,
                write_to_tensorboard=True,
            )
            eval_duration += timers("eval-time").elapsed()
            eval_iterations += train_config.eval_iters
            timers("eval-time").stop()

            if train_config.manual_gc and train_config.manual_gc_eval:
                # Collect only the objects created and used in evaluation.
                gc.collect(generation=0)
            timers("interval-time", log_level=0).start(barrier=True)

        # Miscellaneous post-training-step functions (e.g., FT heartbeats, GC).
        # Some of these only happen at specific iterations.
        post_training_step_callbacks(
            global_state.train_state.step,
            prof,
            config,
        )

        # Checkpoint and decide whether to exit.
        should_exit = checkpoint_and_decide_exit(
            global_state,
            model,
            optimizer,
            scheduler,
        )
        if should_exit:
            break

    # Flush TensorBoard, WandB writers and one-logger.
    writer = global_state.tensorboard_logger
    if writer:
        writer.flush()

    # This will finalize all unfinalized async request and terminate
    # a persistent async worker if persistent ckpt worker is enabled
    # fault_tolerance.on_checkpointing_start(global_state)
    # maybe_finalize_async_save(ckpt_cfg=config.checkpoint_config, blocking=True, terminate=True)
    # fault_tolerance.on_checkpointing_end(global_state=global_state, is_async_finalization=True)

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if should_exit:
        finish_train(global_state)
        sys.exit(exit_code)


def finish_train(global_state: GlobalState):
    # ckpt_cfg = global_state.cfg.checkpoint_config

    # fault_tolerance.on_checkpointing_start(global_state)
    # maybe_finalize_async_save(blocking=True, terminate=True, ckpt_cfg=ckpt_cfg)
    # fault_tolerance.on_checkpointing_end(global_state=global_state, is_async_finalization=True)
    # fault_tolerance.shutdown(global_state)

    if global_state.wandb_logger:
        global_state.wandb_logger.finish()


def save_checkpoint_and_time(
    state: GlobalState,
    model,
    optimizer,
    opt_param_scheduler,
):
    timers = state.timers

    # Stop timer to get accurate train interval time and exclude checkpointing duration
    timers("interval-time").stop()
    # Extra barrier is added to make sure all ranks report the max time.
    timer_key = "save-checkpoint"
    timers(timer_key, log_level=0).start(barrier=True)

    save_checkpoint(
        state.cfg.checkpoint_config.save,
        state,
        model,
        optimizer,
        opt_param_scheduler,
        state.cfg.dataset_config.tokenizer,
        save_rng=state.cfg.checkpoint_config.save_rng,
        save_optim=state.cfg.checkpoint_config.save_optim,
    )
    timers(timer_key).stop(barrier=True)
    timers.log([timer_key])

    if state.cfg.logger_config.log_progress:
        compute_throughputs_and_append_to_progress_log(state)

    # Recover timing
    timers("interval-time", log_level=0).start(barrier=True)


def checkpoint_and_decide_exit(
    state: GlobalState,
    model,
    optimizer,
    opt_param_scheduler,
):
    """Save checkpoint and decide whether to exit based on arguments (e.g., if
    --exit-duration-in-mins is set). Actual exit happens in main training loop
    based on the return value of this function."""
    saved_checkpoint = False

    # Exit based on signal handler.
    if state.cfg.train_config.exit_signal_handler:
        signal_handler = state.signal_handler
        if any(signal_handler.signals_received()):
            if state.cfg.checkpoint_config.save:
                save_checkpoint_and_time(
                    state,
                    model,
                    optimizer,
                    opt_param_scheduler,
                )
            barrier_and_log("exiting program after receiving SIGTERM.")

            return True

    # Regular save (persistent).
    if (
        state.cfg.checkpoint_config.save
        and state.cfg.checkpoint_config.save_interval
        and state.train_state.step % state.cfg.checkpoint_config.save_interval == 0
    ):
        save_checkpoint_and_time(
            state,
            model,
            optimizer,
            opt_param_scheduler,
        )
        saved_checkpoint = True

    # Exit based on duration.
    if state.cfg.train_config.exit_duration_in_mins:
        train_time = (time.time() - state.train_state.start_time) / 60.0
        done_cuda = torch.tensor(
            [train_time > state.cfg.checkpoint_config.exit_duration_in_mins], dtype=torch.int, device="cuda"
        )
        torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
        done = done_cuda.item()
        if done:
            if state.cfg.checkpoint_config.save and not saved_checkpoint:
                save_checkpoint_and_time(
                    state,
                    model,
                    optimizer,
                    opt_param_scheduler,
                )
            barrier_and_log(f"exiting program after {train_time} minutes")

            return True

    # Exit based on iterations.
    if state.cfg.train_config.exit_interval and state.train_state.step % state.cfg.train_config.exit_interval == 0:
        if state.cfg.checkpoint_config.save and not saved_checkpoint:
            save_checkpoint_and_time(
                state,
                model,
                optimizer,
                opt_param_scheduler,
            )
        barrier_and_log(f"exiting program at iteration {state.train_state.step}")

        return True

    return False
