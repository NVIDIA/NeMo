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

import inspect
from functools import partial
from typing import Callable, Optional

from nemo.automodel.checkpointing import save_checkpoint
from nemo.automodel.config import ConfigContainer
from nemo.automodel.eval import evaluate_and_print_results
from nemo.automodel.setup import setup
from nemo.automodel.train import finish_train, train
from nemo.tron.data.utils import get_dataset_provider
from nemo.tron.utils.common_utils import barrier_and_log, print_rank_0


def automodel_train(
    config: ConfigContainer,
    forward_step_func: Callable,
    dataset_provider: Optional[Callable] = None,
):
    config.validate()

    ## SETUP ##
    if dataset_provider is None:
        dataset_provider = get_dataset_provider(config.dataset_config)

    setup_output = setup(config, dataset_provider)
    state = setup_output.state
    model = setup_output.model
    optimizer = setup_output.optimizer
    scheduler = setup_output.scheduler
    train_data_iterator = setup_output.train_data_iterator
    valid_data_iterator = setup_output.valid_data_iterator
    test_data_iterator = setup_output.test_data_iterator

    if "state" in inspect.signature(forward_step_func).parameters:
        wrapped_forward_step = partial(forward_step_func, state=state)
    else:
        wrapped_forward_step = forward_step_func

    ## TRAINING ##
    if not config.train_config.skip_train:
        barrier_and_log("training ...")
        if state.train_state.do_train and config.train_config.train_iters > 0:
            train(
                forward_step_func=wrapped_forward_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_data_iterator=train_data_iterator,
                valid_data_iterator=valid_data_iterator,
                global_state=state,
            )

        barrier_and_log("after training is done")
        ckpt_config = config.checkpoint_config
        if ckpt_config.save and state.train_state.step != 0 and ckpt_config.save_interval != 0:
            save_checkpoint(
                save_dir=ckpt_config.save,
                state=state,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                tokenizer=config.dataset_config.tokenizer,
                save_rng=ckpt_config.save_rng,
                save_optim=ckpt_config.save_optim,
            )
    else:
        barrier_and_log("skipping training ...")

    iteration = state.train_state.step
    ## VALIDATION ##
    if state.train_state.do_valid:
        prefix = f"iteration {iteration} on validation set"
        evaluate_and_print_results(
            state,
            prefix,
            wrapped_forward_step,
            valid_data_iterator,
            model,
            config.model_config,
            write_to_tensorboard=not config.train_config.skip_train,
        )
    if state.train_state.do_test:
        prefix = f"iteration {iteration} on test set"
        evaluate_and_print_results(
            state,
            prefix,
            wrapped_forward_step,
            test_data_iterator,
            model,
            config.model_config,
            write_to_tensorboard=not config.train_config.skip_train,
        )

    finish_train(state)
