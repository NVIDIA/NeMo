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

from typing import Callable, Optional

from nemo.tron.checkpointing import save_checkpoint
from nemo.tron.config import ConfigContainer
from nemo.tron.eval import evaluate_and_print_results
from nemo.tron.setup import setup
from nemo.tron.train import _finish_train, train
from nemo.tron.utils.common_utils import barrier_and_log, print_rank_0


def megatron_pretrain(
    forward_step_func: Callable,
    full_config: ConfigContainer,
    dataset_provider: Callable,
    get_embedding_ranks: Optional[Callable] = None,
    get_position_embedding_ranks: Optional[Callable] = None,
    process_non_loss_data_func: Optional[Callable] = None,
    non_loss_data_func: Optional[Callable] = None,
):
    ## SETUP ##
    setup_output = setup(full_config, dataset_provider, get_embedding_ranks, get_position_embedding_ranks)
    state = setup_output.state
    model = setup_output.model
    optimizer = setup_output.optimizer
    scheduler = setup_output.scheduler
    train_data_iterator = setup_output.train_data_iterator
    valid_data_iterator = setup_output.valid_data_iterator
    test_data_iterator = setup_output.test_data_iterator
    ckpt_context = setup_output.checkpointing_context

    ## TRAINING ##
    if not full_config.megatron_lm_config.skip_train:
        print_rank_0("training ...")
        iteration = state.train_state.step
        if state.train_state.do_train and full_config.megatron_lm_config.train_iters > 0:
            iteration, num_floating_point_operations_so_far = train(
                forward_step_func,
                model,
                optimizer,
                scheduler,
                train_data_iterator,
                valid_data_iterator,
                process_non_loss_data_func,
                state,
                ckpt_context,
                non_loss_data_func,
            )

        barrier_and_log("after training is done")
        ckpt_config = full_config.checkpoint_config
        if ckpt_config.save and state.train_state.step != 0 and ckpt_config.save_interval != 0:
            save_checkpoint(
                state,
                model,
                optimizer,
                scheduler,
                num_floating_point_operations_so_far,
                full_config,
                ckpt_context,
                train_data_iterator=train_data_iterator,
            )

    else:
        print_rank_0("skipping training ...")
        iteration = state.train_state.step

    ## VALIDATION ##
    if state.train_state.do_valid:
        prefix = f"iteration {iteration} on validation set"
        evaluate_and_print_results(
            state,
            prefix,
            forward_step_func,
            valid_data_iterator,
            model,
            process_non_loss_data_func,
            full_config.model_config,
            verbose=True,
            write_to_tensorboard=not full_config.megatron_lm_config.skip_train,
            non_loss_data_func=non_loss_data_func,
        )
    if state.train_state.do_test:
        prefix = f"iteration {iteration} on test set"
        evaluate_and_print_results(
            state,
            prefix,
            forward_step_func,
            test_data_iterator,
            model,
            process_non_loss_data_func,
            full_config.model_config,
            verbose=True,
            write_to_tensorboard=not full_config.megatron_lm_config.skip_train,
            non_loss_data_func=non_loss_data_func,
        )

    _finish_train(full_config.checkpoint_config, state)
