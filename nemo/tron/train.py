import gc
import sys
from functools import partial
from typing import Callable

import torch
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.num_microbatches_calculator import (
    get_current_global_batch_size,
    get_current_running_global_batch_size,
    get_num_microbatches,
    update_num_microbatches,
)
from megatron.core.utils import check_param_hashes_across_dp_replicas

from nemo.tron.config import ConfigContainer, MegatronLMConfig
from nemo.tron.state import GlobalState, TrainState
from nemo.tron.train_utils import calc_params_l2_norm
from nemo.tron.utils import print_rank_0


def forward_step(data_iterator, loss_func, data_step: Callable, model, global_state: GlobalState):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = global_state.timers

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    model_inputs = data_step(data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(**model_inputs)

    return output_tensor, partial(loss_func, model_inputs.get("loss_mask", None))


def _train(
    forward_step_func,
    model,
    optimizer,
    scheduler,
    train_data_iterator,
    valid_data_iterator,
    global_state: GlobalState,
    checkpointing_context,
):
    train_state: TrainState = global_state.train_state
    config: ConfigContainer = global_state.cfg
    model_config = config.model_config
    mlm_config = config.megatron_lm_config
    timers = global_state.timers

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = train_state.step

    model_config.grad_scale_func = optimizer.scale_loss

    ddp_config = config.ddp_config
    if isinstance(model[0], DDP) and ddp_config.overlap_grad_reduce:
        assert model_config.no_sync_func is None, (
            'When overlap_grad_reduce is True, config.no_sync_func must be None; '
            'a custom no_sync_func is not supported when overlapping grad-reduce'
        )
        model_config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            model_config.no_sync_func = model_config.no_sync_func[0]
        if mlm_config.align_grad_reduce:
            model_config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                model_config.grad_sync_func = model_config.grad_sync_func[0]
    if ddp_config.overlap_param_gather and ddp_config.align_param_gather:
        model_config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            model_config.param_sync_func = model_config.param_sync_func[0]
    model_config.finalize_model_grads_func = finalize_model_grads

    timers('interval-time', log_level=0).start(barrier=True)
    report_memory_flag = True
    pre_hook_enabled = False
    should_exit = False
    exit_code = 0

    if mlm_config.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert (
            mlm_config.manual_gc_interval >= 0
        ), 'Manual garbage collection interval should be larger than or equal to 0'
        gc.disable()
        gc.collect()

    num_microbatches = get_num_microbatches()
    eval_duration = 0.0
    eval_iterations = 0

    prof = _profiler_setup(mlm_config)

    start_iteration = iteration
    should_toggle_forward_pre_hook = (
        config.optimizer_config.use_distributed_optimizer and ddp_config.overlap_param_gather
    )
    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if should_toggle_forward_pre_hook:
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = model_config.param_sync_func
        model_config.param_sync_func = None
        pre_hook_enabled = False
    # Also, check weight hash across DP replicas to be very pedantic.
    if mlm_config.check_weight_hash_across_dp_replicas_interval is not None:
        assert check_param_hashes_across_dp_replicas(
            model, cross_check=True
        ), "Parameter hashes not matching across DP replicas"
        torch.distributed.barrier()
        print_rank_0(f">>> Weight hashes match after {iteration} iterations...")

    # Run training iterations till done.
    while iteration < mlm_config.train_iters:
        if mlm_config.profile and torch.distributed.get_rank() in mlm_config.profile_ranks:
            if mlm_config.use_pytorch_profiler:
                prof.step()
            elif iteration == mlm_config.profile_step_start:
                torch.cuda.cudart().cudaProfilerStart()
                torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        update_num_microbatches(train_state.consumed_train_samples, consistency_check=False, verbose=True)
        if get_num_microbatches() != num_microbatches and iteration != 0:
            assert get_num_microbatches() > num_microbatches, (
                f"Number of microbatches should be increasing due to batch size rampup; "
                f"instead going from {num_microbatches} to {get_num_microbatches()}"
            )
            if mlm_config.save is not None:
                save_checkpoint_and_time(
                    iteration,
                    model,
                    optimizer,
                    scheduler,
                    # num_floating_point_operations_so_far,
                    checkpointing_context,
                    train_data_iterator=train_data_iterator,
                )  # TODO (ananth/hemild): implement
        num_microbatches = get_num_microbatches()
        update_num_microbatches(train_state.consumed_train_samples, consistency_check=True, verbose=True)

        # Run training step.
        loss_dict, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad = train_step(
            forward_step_func, iteration, train_data_iterator, model, optimizer, scheduler, config
        )
        if should_checkpoint:
            save_checkpoint_and_time(
                iteration,
                model,
                optimizer,
                scheduler,
                # num_floating_point_operations_so_far,
                checkpointing_context,
                train_data_iterator=train_data_iterator,
            )
        if should_exit:
            break

        # Enable forward pre-hooks after first set of forward and backward passes.
        # When running in fp16, skip all NaN iterations until steady-state loss scaling value
        # is reached.
        if iteration == start_iteration:
            if skipped_iter:
                # Only enable forward pre-hook after a training step has successfully run. Relevant
                # for fp16 codepath where first XX iterations are skipped until steady-state loss
                # scale value is reached.
                start_iteration = iteration + 1
            else:
                # Enable forward pre-hook after training step has successfully run. All subsequent
                # forward passes will use the forward pre-hook / `param_sync_func` in
                # `forward_backward_func`.
                if should_toggle_forward_pre_hook:
                    enable_forward_pre_hook(model)
                    model_config.param_sync_func = param_sync_func
                    pre_hook_enabled = True

        iteration += 1
        batch_size = (
            parallel_state.get_data_parallel_world_size() * mlm_config.micro_batch_size * get_num_microbatches()
        )
        train_state.consumed_train_samples += batch_size
        num_skipped_samples_in_batch = get_current_global_batch_size() - get_current_running_global_batch_size()
        if mlm_config.decrease_batch_size_if_needed:
            assert num_skipped_samples_in_batch >= 0
        else:
            assert num_skipped_samples_in_batch == 0
        train_state.skipped_train_samples += num_skipped_samples_in_batch

        # Logging.
        if not optimizer.is_stub_optimizer:
            loss_scale = optimizer.get_loss_scale().item()
        else:
            loss_scale = 1.0
        params_norm = None

        if mlm_config.log_params_norm:
            params_norm = calc_params_l2_norm(model, model_config)
        learning_rate = None
        decoupled_learning_rate = None
        for param_group in optimizer.param_groups:
            if param_group['is_decoupled_lr']:
                decoupled_learning_rate = param_group['lr']
            else:
                learning_rate = param_group['lr']
        report_memory_flag = training_log(
            loss_dict,
            total_loss_dict,
            learning_rate,
            decoupled_learning_rate,
            iteration,
            loss_scale,
            report_memory_flag,
            skipped_iter,
            grad_norm,
            params_norm,
            num_zeros_in_grad,
        )  # TODO: implement

        if train_state.do_valid and mlm_config.eval_interval and iteration % mlm_config.eval_interval == 0:
            timers('interval-time').stop()
            if should_toggle_forward_pre_hook:
                disable_forward_pre_hook(model)
                pre_hook_enabled = False
            if mlm_config.manual_gc and mlm_config.manual_gc_eval:
                # Collect all objects.
                gc.collect()
            prefix = f'iteration {iteration}'
            timers('eval-time', log_level=0).start(barrier=True)
            evaluate_and_print_results(
                prefix,
                forward_step_func,
                valid_data_iterator,
                model,
                iteration,
                # process_non_loss_data_func,
                config,
                verbose=False,
                write_to_tensorboard=True,
                # non_loss_data_func=non_loss_data_func
            )  # TODO (hemild): implement
            eval_duration += timers('eval-time').elapsed()
            eval_iterations += mlm_config.eval_iters
            timers('eval-time').stop()

            if mlm_config.manual_gc and mlm_config.manual_gc_eval:
                # Collect only the objects created and used in evaluation.
                gc.collect(generation=0)
            if should_toggle_forward_pre_hook:
                enable_forward_pre_hook(model)
                pre_hook_enabled = True
            timers('interval-time', log_level=0).start(barrier=True)

        # Miscellaneous post-training-step functions (e.g., FT heartbeats, GC).
        # Some of these only happen at specific iterations.
        post_training_step_callbacks(
            model,
            optimizer,
            scheduler,
            iteration,
            prof,
            # num_floating_point_operations_since_last_log_event,
            should_toggle_forward_pre_hook,
        )

        # Checkpoint and decide whether to exit.
        should_exit = checkpoint_and_decide_exit(
            model,
            optimizer,
            scheduler,
            iteration,
            # num_floating_point_operations_so_far,
            checkpointing_context,
            train_data_iterator,
        )
        if should_exit:
            break

    # Flush TensorBoard, WandB writers and one-logger.
    writer = global_state.tensorboard_logger
    if writer:
        writer.flush()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if should_exit:
        wandb_writer = global_state.wandb_logger
        if wandb_writer:
            wandb_writer.finish()
        sys.exit(exit_code)

    return iteration


def post_training_step_callbacks(
    model, optimizer, scheduler, iteration, prof, mlm_config: MegatronLMConfig, should_toggle_forward_pre_hook: bool
):
    """Run all post-training-step functions (e.g., FT heartbeats, GC)."""

    # Bring CPU and GPU back in sync if on right iteration.
    if mlm_config.train_sync_interval and iteration % mlm_config.train_sync_interval == 0:
        torch.cuda.synchronize()

    # Check weight hash across DP replicas.
    if (
        mlm_config.check_weight_hash_across_dp_replicas_interval is not None
        and iteration % mlm_config.check_weight_hash_across_dp_replicas_interval == 0
    ):
        if should_toggle_forward_pre_hook:
            disable_forward_pre_hook(model)
        assert check_param_hashes_across_dp_replicas(
            model, cross_check=True
        ), "Parameter hashes not matching across DP replicas"
        torch.distributed.barrier()
        print_rank_0(f">>> Weight hashes match after {iteration} iterations...")
        if should_toggle_forward_pre_hook:
            enable_forward_pre_hook(model)

    # Profiling.
    if (
        mlm_config.profile
        and iteration == mlm_config.profile_step_end
        and torch.distributed.get_rank() in mlm_config.profile_ranks
    ):
        if mlm_config.use_pytorch_profiler:
            assert prof is not None
            prof.stop()
        else:
            torch.cuda.cudart().cudaProfilerStop()

    # Manual garbage collection.
    if mlm_config.manual_gc:
        if mlm_config.manual_gc_interval != 0 and iteration % mlm_config.manual_gc_interval == 0:
            gc.collect()


def _profiler_setup(config: MegatronLMConfig):
    prof = None
    if config.profile and torch.distributed.get_rank() in config.profile_ranks and config.use_pytorch_profiler:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=max(config.profile_step_start - 1, 0),
                warmup=1 if config.profile_step_start > 0 else 0,
                active=config.profile_step_end - config.profile_step_start,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(config.tensorboard_dir),
            record_shapes=True,
            with_stack=True,
        )
        prof.start()

    return prof


def enable_forward_pre_hook(model_chunks):
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model_chunks, param_sync=True):
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.disable_forward_pre_hook(param_sync=param_sync)
