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

import contextlib
import gc
import logging
import sys
import time
from typing import Callable, Iterator

import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer

from nemo.automodel.checkpointing import checkpoint_exists, load_checkpoint
from nemo.automodel.config import ConfigContainer
from nemo.automodel.init import initialize_automodel
from nemo.automodel.utils.train_utils import (
    checkpoint_and_decide_exit,
    eval_log,
    reduce_loss,
    save_checkpoint_and_time,
    training_log,
)
from nemo.tron.state import GlobalState
from nemo.tron.utils.common_utils import (
    append_to_progress_log,
    barrier_and_log,
    get_rank_safe,
    get_world_size_safe,
    print_rank_0,
)
from nemo.tron.utils.log_utils import setup_logging

logger = logging.getLogger(__name__)


def setup(
    cfg: ConfigContainer,
):
    state = GlobalState()
    state.cfg = cfg
    # TODO: Freeze state.cfg

    setup_logging(
        logging_level=cfg.logger_config.logging_level,
        filter_warning=cfg.logger_config.filter_warnings,
        modules_to_filter=cfg.logger_config.modules_to_filter,
        set_level_for_all_loggers=cfg.logger_config.set_level_for_all_loggers,
    )

    initialize_automodel(
        dist_config=cfg.dist_config,
        training_config=cfg.train_config,
        data_parallel_size=cfg.data_parallel_size,
        seed=cfg.rng_config.seed,
    )

    timers = state.timers

    if cfg.logger_config.log_progress:
        append_to_progress_log(cfg.checkpoint_config.save, "Starting job")

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    start_time_tensor = torch.tensor([state.start_time], dtype=torch.double, device="cuda")
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    state.start_time = start_time_tensor.item()

    print_rank_0("time to initialize automodel (seconds): {:.3f}".format(time.time() - state.start_time))
    barrier_and_log("after automodel is initialized")

    # Model, optimizer, and learning rate.
    timers("model-and-optimizer-setup", log_level=0).start(barrier=True)
    model = cfg.model_config.setup(
        use_torch_fsdp2=cfg.dist_config.use_torch_fsdp2,
        wrap_with_ddp=get_world_size_safe() > 1 and not cfg.dist_config.use_torch_fsdp2,
        ddp_kwargs=cfg.model_config.ddp_kwargs,
    )

    optimizer = cfg.optimizer_config.setup(model)
    scheduler = cfg.scheduler_config.setup(optimizer, cfg.optimizer_config.lr, cfg.optimizer_config.min_lr)

    # TODO: Add support for mixed precision training
    timers("model-and-optimizer-setup").stop()
    barrier_and_log("after model, optimizer, and learning rate scheduler are built")

    # Load checkpoint if applicable
    if (cfg.checkpoint_config.load is not None or cfg.checkpoint_config.pretrained_checkpoint is not None) and (
        checkpoint_exists(cfg.checkpoint_config.load) or checkpoint_exists(cfg.checkpoint_config.pretrained_checkpoint)
    ):
        timers("load-checkpoint", log_level=0).start(barrier=True)

        load_checkpoint(
            state,
            model,
            optimizer,
            scheduler,
        )
        timers("load-checkpoint").stop(barrier=True)
        timers.log(["load-checkpoint"])

    # Data stuff.
    timers("train/valid/test-data-iterators-setup", log_level=0).start(barrier=True)

    train_data_iterator, valid_data_iterator, test_data_iterator = cfg.dataset_config.build_iterators(
        rank=get_rank_safe(),
        world_size=get_world_size_safe(),
        micro_batch_size=cfg.train_config.micro_batch_size,
    )
    # Flags to know if we need to do training/validation/testing
    do_train = train_data_iterator is not None and cfg.train_config.train_iters > 0
    do_valid = valid_data_iterator is not None and cfg.train_config.eval_iters > 0
    do_test = test_data_iterator is not None and cfg.train_config.eval_iters > 0
    flags = torch.tensor([int(do_train), int(do_valid), int(do_test)], dtype=torch.long, device="cuda")

    torch.distributed.broadcast(flags, 0)

    train_state = state.train_state
    train_state.do_train = train_state.do_train or flags[0].item()
    train_state.do_valid = train_state.do_valid or flags[1].item()
    train_state.do_test = train_state.do_test or flags[2].item()

    # Log info about dataset configuration
    print_rank_0(f"Training data ready: {do_train}")
    print_rank_0(f"Validation data ready: {do_valid}")
    print_rank_0(f"Test data ready: {do_test}")

    timers("train/valid/test-data-iterators-setup").stop()
    barrier_and_log("after dataloaders are built")

    # Print setup timing.
    print_rank_0("done with setup ...")
    timers.log(["model-and-optimizer-setup", "train/valid/test-data-iterators-setup"], barrier=True)

    return (
        state,
        model,
        optimizer,
        scheduler,
        train_data_iterator,
        valid_data_iterator,
        test_data_iterator,
    )


def forward_with_loss(
    state: GlobalState,
    data_iterator: Iterator,
    model: torch.nn.Module,
    loss_fn: Callable,
):
    timers = state.timers
    straggler_timer = state.straggler_timer

    timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        data = next(data_iterator)
        batch = {key: value.cuda(non_blocking=True) for key, value in data.items()}

    timers("batch-generator").stop()

    # TODO(@boxiangw): Refractor. Needed for SP support
    batch["position_ids"] = torch.arange(0, batch["input_ids"].shape[1]).unsqueeze(0).cuda(non_blocking=True)

    # batch = _remove_extra_batch_keys(batch)
    labels = batch.pop("labels")
    loss_mask = batch.pop("loss_mask", None)
    assert loss_mask is not None, "loss_mask is required for training"

    with straggler_timer:
        outputs = model.forward(**batch)
        # Prepare for loss calculation
        logits = outputs.logits
        n_cls = logits.shape[-1]
        logits = logits.view(-1, n_cls)
        labels = labels.view(-1)
        assert logits.shape[-2] == labels.shape[-1], "Expected logits & labels to have the same length"
        loss = loss_fn(logits, labels, loss_mask)

    return loss, loss_mask.sum()


def train_step(
    data_iterator: Iterator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    num_microbatches: int,
    global_state: GlobalState,
):
    """Single training step."""
    cfg: ConfigContainer = global_state.cfg
    timers = global_state.timers
    train_config = cfg.train_config
    optim_config = cfg.optimizer_config
    model_config = cfg.model_config

    optimizer.zero_grad()
    if hasattr(model, "zero_grad"):
        model.zero_grad()

    # Forward pass.
    timers("forward-backward", log_level=1).start(barrier=model_config.barrier_with_L1_time)

    if isinstance(model, DistributedDataParallel):
        no_sync_func = model.no_sync
    else:
        no_sync_func = contextlib.nullcontext

    forward_data_store = []
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
    with no_sync_func():
        for i in range(num_microbatches - 1):
            loss, num_tokens = forward_with_loss(
                state=global_state, data_iterator=data_iterator, model=model, loss_fn=model_config.loss_fn
            )
            num_tokens = num_tokens.clone().detach().to(torch.int)
            total_num_tokens += num_tokens.item()
            forward_data_store.append(loss.clone())
            loss.backward()

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    loss, num_tokens = forward_with_loss(
        state=global_state, data_iterator=data_iterator, model=model, loss_fn=model_config.loss_fn
    )
    num_tokens = num_tokens.clone().detach().to(torch.int)
    total_num_tokens += num_tokens.item()
    forward_data_store.append(loss.clone())

    loss.backward()

    if model_config.calculate_per_token_loss and total_num_tokens is not None:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism and layernorm all-reduce for sequence parallelism).
        group = None
        # This is the size of the data parallel group, since DDP all reduces grads via averaging
        group_size = get_world_size_safe()
        # TODO: Add support to select data parallel group for FSDP etc

        num_tokens_for_grad_scaling = total_num_tokens.clone().detach()
        torch.distributed.all_reduce(num_tokens_for_grad_scaling, group=group)
        scaling_factor = group_size / num_tokens_for_grad_scaling
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(scaling_factor)

    if timers is not None:
        timers("forward-backward").stop()
    reporting_loss, reporting_num_tokens = reduce_loss(
        forward_data_store, total_num_tokens, per_token_loss=model_config.calculate_per_token_loss
    )

    # Empty unused memory.
    if train_config.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=optim_config.clip_grad)

    # Update parameters.
    timers("optimizer", log_level=1).start(barrier=optim_config.barrier_with_L1_time)
    optimizer.step()
    timers("optimizer").stop()

    # Update learning rate.
    increment = num_microbatches * train_config.micro_batch_size * cfg.data_parallel_size
    scheduler.step(increment=increment)

    # Empty unused memory.
    if train_config.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    return (
        {"lm loss": (reporting_loss, reporting_num_tokens)},
        grad_norm,
    )


def evaluate_and_print_results(
    global_state: GlobalState,
    prefix: str,
    data_iterator,
    model: torch.nn.Module,
    verbose=False,
):
    """Helper function to evaluate and dump results on screen."""
    config: ConfigContainer = global_state.cfg
    model_config = config.model_config
    timers = global_state.timers
    train_config = config.train_config

    timers("evaluate", log_level=0).start(barrier=True)

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss_dict = {}

    # Make validation batch size independent from training batch size
    eval_batch_size = train_config.global_batch_size
    eval_num_microbatches = eval_batch_size // (train_config.micro_batch_size * config.data_parallel_size)

    with torch.no_grad():
        iteration = 0
        if verbose:
            print_rank_0(f"Evaluating on {train_config.eval_iters * eval_batch_size} samples")
        while iteration < train_config.eval_iters:
            iteration += 1
            if verbose:
                print_rank_0(f"Evaluating iter {iteration}/{train_config.eval_iters}")

            loss_store = []
            total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
            for _ in range(eval_num_microbatches):
                loss, num_tokens = forward_with_loss(
                    state=global_state, data_iterator=data_iterator, model=model, loss_fn=model_config.loss_fn
                )
                num_tokens = num_tokens.clone().detach().to(torch.int)
                total_num_tokens += num_tokens.item()
                loss_store.append(loss.clone())

            # Empty unused memory
            if train_config.empty_unused_memory_level >= 1:
                torch.cuda.empty_cache()

            # Reduce across processes.
            loss, total_num_tokens = reduce_loss(loss_store, total_num_tokens)

            if "lm loss" not in total_loss_dict:
                total_loss_dict["lm loss"] = (0.0, 0)

            total_loss_dict["lm loss"] = (
                total_loss_dict["lm loss"][0] + loss.item(),
                total_loss_dict["lm loss"][1] + total_num_tokens.item(),
            )

            global_state.train_state.consumed_valid_samples += eval_batch_size

    # Move model back to the train mode.
    model.train()

    # Calculate average loss
    for key in total_loss_dict:
        numerator, denominator = total_loss_dict[key]
        total_loss_dict[key] = torch.tensor(numerator / denominator, device="cuda")

    timers("evaluate").stop()
    timers.log(["evaluate"])

    eval_log(prefix, total_loss_dict, global_state)


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


def train(
    model,
    optimizer,
    scheduler,
    tokenizer,
    train_data_iterator,
    valid_data_iterator,
    num_microbatches: int,
    global_state: GlobalState,
):
    config: ConfigContainer = global_state.cfg
    train_config = config.train_config
    timers = global_state.timers

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss_dict = {}

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

        # Run training step.
        loss_dict, grad_norm = train_step(
            train_data_iterator, model, optimizer, scheduler, num_microbatches, global_state
        )

        global_state.train_state.step += 1
        batch_size = config.data_parallel_size * train_config.micro_batch_size * num_microbatches
        global_state.train_state.consumed_train_samples += batch_size

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
                valid_data_iterator,
                model,
                verbose=False,
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
            tokenizer,
        )
        if should_exit:
            break

    if (
        config.checkpoint_config.save
        and global_state.train_state.step != 0
        and config.checkpoint_config.save_interval != 0
    ):
        # Save final checkpoint
        save_checkpoint_and_time(
            global_state,
            model,
            optimizer,
            scheduler,
            tokenizer,
        )

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if should_exit:
        # Flush TensorBoard, WandB writers and one-logger.
        if global_state.tensorboard_logger:
            global_state.tensorboard_logger.flush()

        if global_state.wandb_logger:
            global_state.wandb_logger.finish()

        sys.exit(exit_code)


def finetune(
    config: ConfigContainer,
    tokenizer: AutoTokenizer,
):
    config.validate()

    ## SETUP ##
    state, model, optimizer, scheduler, train_data_iterator, valid_data_iterator, test_data_iterator = setup(config)
    num_microbatches_per_step = config.train_config.global_batch_size // (
        config.train_config.micro_batch_size * config.data_parallel_size
    )

    ## TRAINING ##
    if not config.train_config.skip_train:
        barrier_and_log("training ...")
        if state.train_state.do_train and config.train_config.train_iters > 0:
            train(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                tokenizer=tokenizer,
                train_data_iterator=train_data_iterator,
                valid_data_iterator=valid_data_iterator,
                num_microbatches=num_microbatches_per_step,
                global_state=state,
            )

        barrier_and_log("after training is done")
    else:
        barrier_and_log("skipping training ...")

    iteration = state.train_state.step

    ## VALIDATION ##
    if state.train_state.do_valid:
        prefix = f"iteration {iteration} on validation set"
        evaluate_and_print_results(
            state,
            prefix,
            valid_data_iterator,
            model,
        )
    if state.train_state.do_test:
        prefix = f"iteration {iteration} on test set"
        evaluate_and_print_results(
            state,
            prefix,
            test_data_iterator,
            model,
        )

    ## CLEANUP ##
    if state.tensorboard_logger:
        state.tensorboard_logger.flush()

    if state.wandb_logger:
        state.wandb_logger.finish()
