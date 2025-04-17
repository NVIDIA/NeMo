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

import math
import time

import torch

from nemo.automodel.config import ConfigContainer
from nemo.automodel.schedules import ForwardStepFnProtocol, get_forward_backward_func
from nemo.automodel.utils.train_utils import reduce_loss
from nemo.tron.state import GlobalState
from nemo.tron.utils.common_utils import is_last_rank, print_rank_0, print_rank_last


def evaluate(
    forward_step_func: ForwardStepFnProtocol,
    data_iterator,
    model: torch.nn.Module,
    global_state: GlobalState,
    verbose=False,
):
    """Evaluation."""
    config: ConfigContainer = global_state.cfg
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

            forward_backward_func = get_forward_backward_func()
            loss_store, total_num_tokens = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                config=config.model_config,
                num_microbatches=eval_num_microbatches,
                forward_only=True,
                timers=timers,
            )

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

            if train_config.exit_duration_in_mins:
                train_time = (time.time() - global_state.start_time) / 60.0
                done_cuda = torch.tensor(
                    [train_time > train_config.exit_duration_in_mins], dtype=torch.int, device="cuda"
                )
                torch.distributed.all_reduce(done_cuda, op=torch.distributed.ReduceOp.MAX)
                done = done_cuda.item()
                if done:
                    print_rank_0("Exiting during evaluation, timelimit reached")
                    return None, None, True

    # Move model back to the train mode.
    model.train()

    # Calculate average loss
    for key in total_loss_dict:
        numerator, denominator = total_loss_dict[key]
        total_loss_dict[key] = torch.tensor(numerator / denominator, device="cuda")

    timers("evaluate").stop()
    timers.log(["evaluate"])

    return total_loss_dict, False


def evaluate_and_print_results(
    global_state: GlobalState,
    prefix: str,
    forward_step_func: ForwardStepFnProtocol,
    data_iterator,
    model: torch.nn.Module,
    verbose=False,
    write_to_tensorboard=True,
):
    """Helper function to evaluate and dump results on screen."""
    if write_to_tensorboard:
        writer = global_state.tensorboard_logger
    else:
        writer = None

    wandb_writer = global_state.wandb_logger

    total_loss_dict, timelimit = evaluate(forward_step_func, data_iterator, model, global_state, verbose)

    # Timelimit hit during evaluation
    if timelimit:
        return

    string = f" validation loss at {prefix} | "
    for key in total_loss_dict:
        string += "{} value: {:.6E} | ".format(key, total_loss_dict[key].item())
        ppl = math.exp(min(20, total_loss_dict[key].item()))
        string += "{} PPL: {:.6E} | ".format(key, ppl)
        if writer:
            writer.add_scalar("{} validation".format(key), total_loss_dict[key].item(), global_state.train_state.step)
            writer.add_scalar(
                "{} validation vs samples".format(key),
                total_loss_dict[key].item(),
                global_state.train_state.consumed_train_samples,
            )
            if global_state.cfg.logger_config.log_validation_ppl_to_tensorboard:
                writer.add_scalar("{} validation ppl".format(key), ppl, global_state.train_state.step)
                writer.add_scalar(
                    "{} validation ppl vs samples".format(key), ppl, global_state.train_state.consumed_train_samples
                )

        if wandb_writer and is_last_rank():
            wandb_writer.log({"{} validation".format(key): total_loss_dict[key].item()}, global_state.train_state.step)
            if global_state.cfg.logger_config.log_validation_ppl_to_tensorboard:
                wandb_writer.log({"{} validation ppl".format(key): ppl}, global_state.train_state.step)

    length = len(string) + 1
    print_rank_last("-" * length)
    print_rank_last(string)
    print_rank_last("-" * length)
