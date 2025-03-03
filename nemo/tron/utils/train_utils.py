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

from datetime import datetime

import torch
from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.core.utils import get_data_parallel_group_if_dtensor, to_local_if_dtensor

from nemo.tron.config import ConfigContainer
from nemo.tron.state import GlobalState
from nemo.tron.utils.common_utils import get_world_size_safe, is_last_rank, print_rank_last
from nemo.tron.utils.theoretical_memory_utils import report_theoretical_memory

try:
    from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm
except ImportError:
    try:
        from amp_C import multi_tensor_l2norm
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        import warnings

        warnings.warn(
            "Transformer Engine and Apex are not installed. "
            "Falling back to local implementations of "
            "multi_tensor_applier and multi_tensor_l2norm"
        )

        from megatron.core.utils import local_multi_tensor_applier as multi_tensor_applier
        from megatron.core.utils import local_multi_tensor_l2_norm as multi_tensor_l2norm


def param_is_not_shared(param):
    return not hasattr(param, "shared") or not param.shared


def calc_params_l2_norm(model, model_config, force_create_fp32_copy=False):
    """Calculate l2 norm of parameters"""
    if not isinstance(model, list):
        model = [model]
    # Seperate moe and dense params
    params_data = []
    moe_params_data = []
    sharded_params_data = []
    data_parallel_group = None

    for model_chunk in model:
        for param in model_chunk.parameters():
            data_parallel_group = get_data_parallel_group_if_dtensor(param, data_parallel_group)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if not is_not_tp_duplicate:
                continue
            assert is_not_tp_duplicate
            if not getattr(param, "allreduce", True):
                # TODO: Implement memory optimization for MoE parameters.
                assert param_is_not_shared(param)
                param = to_local_if_dtensor(param)
                moe_params_data.append(param.data.float() if model_config.bf16 else param.data)
            else:
                if param_is_not_shared(param):
                    param = to_local_if_dtensor(param)
                    if model_config.bf16:
                        if not force_create_fp32_copy and hasattr(param, "main_param"):
                            if getattr(param, "main_param_sharded", False):
                                if param.main_param is not None:
                                    sharded_params_data.append(param.main_param)
                            else:
                                params_data.append(param.main_param)
                        else:
                            # Fallback to original logic of making a fp32 copy of the
                            # parameter if `.main_param` attribute is not available.
                            params_data.append(param.data.float())
                    else:
                        params_data.append(param.data)

    # Calculate norm.
    dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
    if len(params_data) > 0:
        norm, _ = multi_tensor_applier(
            multi_tensor_l2norm,
            dummy_overflow_buf,
            [params_data],
            False,  # no per-parameter norm.
        )
        norm_2 = norm * norm
    else:
        norm_2 = torch.zeros((1,), dtype=torch.float32, device="cuda")

    if data_parallel_group is not None:
        torch.distributed.all_reduce(norm_2, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group)

    # Add norm contribution from params with sharded main_params. These norms need to be
    # accumulated across the DP group since the main parameters are sharded because
    # of distributed optimizer.
    if len(sharded_params_data) > 0:
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
        sharded_norm, _ = multi_tensor_applier(
            multi_tensor_l2norm,
            dummy_overflow_buf,
            [sharded_params_data],
            False,  # no per-parameter norm.
        )
        sharded_norm_2 = sharded_norm * sharded_norm
        # Sum over all DP groups.
        torch.distributed.all_reduce(
            sharded_norm_2, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_data_parallel_group()
        )
        norm_2 += sharded_norm_2

    # Sum across all model-parallel GPUs (tensor + pipeline).
    torch.distributed.all_reduce(
        norm_2, op=torch.distributed.ReduceOp.SUM, group=parallel_state.get_model_parallel_group()
    )

    # Add norm contribution from expert layers in MoEs.
    if len(moe_params_data) > 0:
        moe_norm, _ = multi_tensor_applier(
            multi_tensor_l2norm,
            dummy_overflow_buf,
            [moe_params_data],
            False,  # no per-parameter norm.
        )
        moe_norm_2 = moe_norm * moe_norm
        # Sum across expert tensor, model and pipeline parallel GPUs.
        torch.distributed.all_reduce(
            moe_norm_2,
            op=torch.distributed.ReduceOp.SUM,
            group=parallel_state.get_expert_tensor_model_pipeline_parallel_group(),
        )
        norm_2 += moe_norm_2

    return norm_2.item() ** 0.5


def reduce_max_stat_across_model_parallel_group(stat: float) -> float:
    """
    Ranks without an optimizer will have no grad_norm or num_zeros_in_grad stats.
    We need to ensure the logging and writer rank has those values.
    This function reduces a stat tensor across the model parallel group.

    We use an all_reduce max since the values have already been summed across optimizer ranks where possible
    """
    if stat is None:
        stat = -1.0
    stat = torch.tensor([stat], dtype=torch.float32, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        stat, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
    )
    if stat.item() == -1.0:
        return None
    else:
        return stat.item()


def logical_and_across_model_parallel_group(input: bool) -> bool:
    """
    This function gathers a bool value across the model parallel group
    """
    if input is True:
        input = 1
    else:
        input = 0
    input = torch.tensor([input], dtype=torch.int, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        input, op=torch.distributed.ReduceOp.MIN, group=parallel_state.get_model_parallel_group()
    )
    return bool(input.item())


def training_log(
    loss_dict,
    total_loss_dict,
    learning_rate,
    decoupled_learning_rate,
    loss_scale,
    report_memory_flag,
    skipped_iter,
    grad_norm,
    params_norm,
    num_zeros_in_grad,
    config: ConfigContainer,
    global_state: GlobalState,
):
    timers = global_state.timers
    train_state = global_state.train_state
    tb_logger = global_state.tensorboard_logger
    wandb_logger = global_state.wandb_logger
    logger_config = config.logger_config
    train_config = config.train_config

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = "advanced iterations"
    skipped_iters_key = "skipped iterations"
    nan_iters_key = "nan iterations"
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(skipped_iters_key, 0) + skipped_iter
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = (
                total_loss_dict.get(key, torch.tensor([0.0], dtype=torch.float, device="cuda")) + loss_dict[key]
            )
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float("inf") or value == -float("inf") or value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        "forward-backward",
        "forward-compute",
        "backward-compute",
        "batch-generator",
        "forward-recv",
        "forward-send",
        "backward-recv",
        "backward-send",
        "forward-send-forward-recv",
        "forward-send-backward-recv",
        "backward-send-forward-recv",
        "backward-send-backward-recv",
        "forward-backward-send-forward-backward-recv",
        "layernorm-grads-all-reduce",
        "embedding-grads-all-reduce",
        "all-grads-sync",
        "params-all-gather",
        "optimizer-copy-to-main-grad",
        "optimizer-unscale-and-check-inf",
        "optimizer-clip-main-grad",
        "optimizer-count-zeros",
        "optimizer-inner-step",
        "optimizer-copy-main-to-model-params",
        "optimizer",
    ]

    # Calculate batch size.
    batch_size = train_config.micro_batch_size * config.data_parallel_size * get_num_microbatches()

    total_iterations = total_loss_dict[advanced_iters_key] + total_loss_dict[skipped_iters_key]

    # learning rate will be None on ranks without trainable params, so we must gather across mp ranks
    learning_rate = reduce_max_stat_across_model_parallel_group(learning_rate)
    # Tensorboard values.
    # Timer requires all the ranks to call.
    if logger_config.log_timers_to_tensorboard and (train_state.step % logger_config.tensorboard_log_interval == 0):
        reset_in_tb = False if hasattr(timers, "write_to_wandb") else True
        timers.write(timers_to_log, tb_logger, train_state.step, normalizer=total_iterations, reset=reset_in_tb)
        if hasattr(timers, "write_to_wandb"):
            timers.write_to_wandb(
                timers_to_log, wandb_logger, train_state.step, normalizer=total_iterations, reset=True
            )

    if tb_logger and (train_state.step % logger_config.tensorboard_log_interval == 0):
        if config.profiling_config:
            if config.profiling_config.record_memory_history and is_last_rank():
                snapshot = torch.cuda.memory._snapshot()
                from pickle import dump

                with open(config.profiling_config.memory_snapshot_path, "wb") as f:
                    dump(snapshot, f)

        if wandb_logger:
            wandb_logger.log({"samples vs steps": global_state.train_state.consumed_train_samples}, train_state.step)
        tb_logger.add_scalar("learning-rate", learning_rate, train_state.step)
        tb_logger.add_scalar(
            "learning-rate vs samples", learning_rate, global_state.train_state.consumed_train_samples
        )
        if wandb_logger:
            wandb_logger.log({"learning-rate": learning_rate}, train_state.step)
        if config.optimizer_config.decoupled_lr is not None:
            tb_logger.add_scalar("decoupled-learning-rate", decoupled_learning_rate, train_state.step)
        if global_state.train_state.skipped_train_samples > 0:
            tb_logger.add_scalar(
                "skipped-train-samples", global_state.train_state.skipped_train_samples, train_state.step
            )
            if wandb_logger:
                wandb_logger.log(
                    {"skipped-train-samples": global_state.train_state.skipped_train_samples}, train_state.step
                )
        tb_logger.add_scalar("batch-size", batch_size, train_state.step)
        tb_logger.add_scalar("batch-size vs samples", batch_size, global_state.train_state.consumed_train_samples)
        if wandb_logger:
            wandb_logger.log({"batch-size": batch_size}, train_state.step)
        for key in loss_dict:
            tb_logger.add_scalar(key, loss_dict[key], train_state.step)
            tb_logger.add_scalar(key + " vs samples", loss_dict[key], global_state.train_state.consumed_train_samples)
            if wandb_logger:
                wandb_logger.log({key: loss_dict[key]}, train_state.step)
        if logger_config.log_loss_scale_to_tensorboard:
            tb_logger.add_scalar("loss-scale", loss_scale, train_state.step)
            tb_logger.add_scalar("loss-scale vs samples", loss_scale, global_state.train_state.consumed_train_samples)
            if wandb_logger:
                wandb_logger.log({"loss-scale": loss_scale}, train_state.step)
        if logger_config.log_world_size_to_tensorboard:
            tb_logger.add_scalar("world-size", get_world_size_safe(), train_state.step)
            tb_logger.add_scalar(
                "world-size vs samples", get_world_size_safe(), global_state.train_state.consumed_train_samples
            )
            if wandb_logger:
                wandb_logger.log({"world-size": get_world_size_safe()}, train_state.step)
        if grad_norm is not None:
            tb_logger.add_scalar("grad-norm", grad_norm, train_state.step)
            tb_logger.add_scalar("grad-norm vs samples", grad_norm, global_state.train_state.consumed_train_samples)
            if wandb_logger:
                wandb_logger.log({"grad-norm": grad_norm}, train_state.step)
        if num_zeros_in_grad is not None:
            tb_logger.add_scalar("num-zeros", num_zeros_in_grad, train_state.step)
            tb_logger.add_scalar(
                "num-zeros vs samples", num_zeros_in_grad, global_state.train_state.consumed_train_samples
            )
            if wandb_logger:
                wandb_logger.log({"num-zeros": num_zeros_in_grad}, train_state.step)
        if params_norm is not None:
            tb_logger.add_scalar("params-norm", params_norm, train_state.step)
            tb_logger.add_scalar(
                "params-norm vs samples", params_norm, global_state.train_state.consumed_train_samples
            )
            if wandb_logger:
                wandb_logger.log({"params-norm": params_norm}, train_state.step)
        if logger_config.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            tb_logger.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                train_state.step,
            )
            tb_logger.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                train_state.step,
            )
            tb_logger.add_scalar(
                "mem-max-allocated-bytes",
                mem_stats["allocated_bytes.all.peak"],
                train_state.step,
            )
            tb_logger.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                train_state.step,
            )
    if config.model_config.num_moe_experts is not None:
        moe_loss_scale = 1 / get_num_microbatches()
        track_moe_metrics(
            moe_loss_scale,
            train_state.step,
            tb_logger,
            wandb_logger,
            total_loss_dict,
            config.model_config.moe_per_layer_logging,
        )

    if train_state.step % logger_config.log_interval == 0:
        elapsed_time = timers("interval-time").elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations

        # throughput = num_floating_point_operations(args, batch_size) / (
        #     elapsed_time_per_iteration * 10**12 * get_world_size_safe())  # TODO: implement

        if logger_config.log_timers_to_tensorboard:
            if tb_logger:
                tb_logger.add_scalar("iteration-time", elapsed_time_per_iteration, train_state.step)
            if wandb_logger:
                wandb_logger.log({"iteration-time": elapsed_time_per_iteration}, train_state.step)
        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += " iteration {:8d}/{:8d} |".format(train_state.step, train_config.train_iters)
        log_string += " consumed samples: {:12d} |".format(global_state.train_state.consumed_train_samples)
        if global_state.train_state.skipped_train_samples > 0:
            log_string += " skipped samples: {:12d} |".format(global_state.train_state.skipped_train_samples)
        log_string += " elapsed time per iteration (ms): {:.1f} |".format(elapsed_time_per_iteration * 1000.0)
        # if logger_config.log_throughput:
        #     log_string += f' throughput per GPU (TFLOP/s/GPU): {throughput:.1f} |'
        #     if logger_config.log_timers_to_tensorboard:
        #         if tb_logger:
        #             tb_logger.add_scalar('throughput', throughput, train_state.step)
        #         if wandb_logger:
        #             wandb_logger.log({'throughput': throughput}, train_state.step) # TODO: enable after flops is implemented
        # Decoupled_learning_rate should be not None only on first and last pipeline stage.
        log_string += f" learning rate: {learning_rate:.6E} |"
        if config.optimizer_config.decoupled_lr is not None and (
            parallel_state.is_pipeline_first_stage(ignore_virtual=True)
            or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
        ):
            assert decoupled_learning_rate is not None
            log_string += f" decoupled learning rate: {decoupled_learning_rate:.6E} |"
        else:
            assert decoupled_learning_rate is None
        log_string += f" global batch size: {batch_size:5d} |"
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key, nan_iters_key]:
                avg = total_loss_dict[key].item() / float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += " {}: {:.6E} |".format(key, avg)
                total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device="cuda")
        log_string += f" loss scale: {loss_scale:.1f} |"
        if grad_norm is not None:
            log_string += f" grad norm: {grad_norm:.3f} |"
        if num_zeros_in_grad is not None:
            log_string += f" num zeros: {num_zeros_in_grad} |"
        if params_norm is not None:
            log_string += f" params norm: {params_norm:.3f} |"
        log_string += " number of skipped iterations: {:3d} |".format(total_loss_dict[skipped_iters_key])
        log_string += " number of nan iterations: {:3d} |".format(total_loss_dict[nan_iters_key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag:
            # Report memory after optimizer state has been initialized.
            if torch.distributed.get_rank() == 0:
                num_microbatches = get_num_microbatches()
                report_theoretical_memory(config, num_microbatches=num_microbatches, verbose=True)
            report_memory(f"(after {train_state.step} iterations)")
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=logger_config.log_interval)

    return report_memory_flag


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + " memory (MB)"
    string += " | allocated: {}".format(torch.cuda.memory_allocated() / mega_bytes)
    string += " | max allocated: {}".format(torch.cuda.max_memory_allocated() / mega_bytes)
    string += " | reserved: {}".format(torch.cuda.memory_reserved() / mega_bytes)
    string += " | max reserved: {}".format(torch.cuda.max_memory_reserved() / mega_bytes)
    if parallel_state.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string), flush=True)


def track_moe_metrics(
    loss_scale, iteration, tb_logger, wandb_logger=None, total_loss_dict=None, per_layer_logging=False
):
    """Track the MoE metrics for logging."""
    # Aux loss logging
    reduce_aux_losses_tracker_across_ranks()
    tracker = parallel_state.get_moe_layer_wise_logging_tracker()
    if tb_logger is not None:
        aux_losses = {k: v["values"].float() * loss_scale for k, v in tracker.items()}
        for name, loss_list in aux_losses.items():
            if total_loss_dict is not None:
                if name not in total_loss_dict:
                    total_loss_dict[name] = loss_list.mean()
                else:
                    total_loss_dict[name] += loss_list.mean()

            # currently when using add_scalars,
            # torch.utils.add_scalars makes each timer its own run, which
            # polutes the runs list, so we just add each as a scalar
            tb_logger.add_scalar(name, loss_list.mean(), iteration)
            if per_layer_logging:
                for i, loss in enumerate(loss_list.tolist()):
                    tb_logger.add_scalar(f"moe/{name}_layer_{i}", loss, iteration)

            # W&B logging lacks support for logging multiple scalars simultaneously.
            # As a workaround, we log each scalar individually first, then we can create
            # a custom panel to manually group them to a single plot.
            if wandb_logger:
                wandb_logger.log({f"{name}": loss_list.mean()}, iteration)
                if per_layer_logging:
                    wandb_logger.log(
                        {f"moe/{name}_layer_{i}": loss for i, loss in enumerate(loss_list.tolist())},
                        iteration,
                    )

    clear_aux_losses_tracker()


def clear_aux_losses_tracker():
    """Clear the auxiliary losses."""
    tracker = parallel_state.get_moe_layer_wise_logging_tracker()
    for name in tracker:
        tracker[name]["values"].zero_()
        tracker[name]["reduce_group"] = None
        tracker[name]["avg_group"] = None


def reduce_aux_losses_tracker_across_ranks():
    """Collect and reduce the auxiliary losses across ranks."""
    tracker = parallel_state.get_moe_layer_wise_logging_tracker()
    for name in tracker:
        values = tracker[name]["values"]
        # Collect aux losses across PP.
        torch.distributed.all_reduce(values, group=parallel_state.get_pipeline_model_parallel_group())
        # Reduce aux losses across ranks.
        if tracker[name].get("reduce_group") is not None:
            torch.distributed.all_reduce(values, group=tracker[name].get("reduce_group"))
        if tracker[name].get("avg_group") is not None:
            torch.distributed.all_reduce(values, group=tracker[name]["avg_group"], op=torch.distributed.ReduceOp.AVG)
