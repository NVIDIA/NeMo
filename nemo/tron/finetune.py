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
import time
from functools import partial
from typing import Callable, Optional

import torch
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
    finalize_model_grads,
)
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.transformer import MegatronModule

from nemo.tron import fault_tolerance
from nemo.tron.checkpointing import checkpoint_exists, load_checkpoint, save_checkpoint
from nemo.tron.config import ConfigContainer
from nemo.tron.data.loaders import setup_data_iterators
from nemo.tron.data.utils import get_dataset_provider
from nemo.tron.eval import evaluate_and_print_results
from nemo.tron.peft import PEFT
from nemo.tron.setup import _init_checkpointing_context, _update_model_config_funcs
from nemo.tron.train import _finish_train, train
from nemo.tron.init import initialize_megatron, set_jit_fusion_options
from nemo.tron.model import get_base_model, get_distributed_model
from nemo.tron.optim import setup_optimizer
from nemo.tron.state import GlobalState
from nemo.tron.tokenizers.tokenizer import build_tokenizer
from nemo.tron.utils.common_utils import append_to_progress_log, barrier_and_log, print_rank_0
from nemo.tron.utils.decorators import experimental_fn
from nemo.tron.utils.log_utils import setup_logging


@experimental_fn
def megatron_peft(
    config: ConfigContainer,
    forward_step_func: Callable,
    peft: PEFT,
    dataset_provider: Optional[Callable] = None,
) -> None:
    """Main function to run the finetuning pipeline.

    Sets up the environment, model, optimizer, scheduler, and data iterators.
    Performs training, validation, and optionally testing based on the provided
    configuration.

    Args:
        config: The main configuration container holding all necessary parameters.
        forward_step_func: A callable that performs a single forward and backward
                           step, returning the loss and any computed metrics.
        peft: PEFT transformation to apply to the model.
        TODO: Make PEFT optional, and unify paths between this and SFT under one common fine-tune API.
        dataset_provider: Optional callable to provide train/validation/test
                          datasets. If None, it's assumed the dataset
                          configuration is self-contained within `config`.

    Warnings:
        This is an experimental API and is subject to change in backwards
        incompatible ways without notice.
    """
    config.validate()
    if not checkpoint_exists(config.checkpoint_config.pretrained_checkpoint):
        raise FileNotFoundError(
            f"Pretrained checkpoint directory not found or invalid: {config.checkpoint_config.pretrained_checkpoint}"
        )

    # SETUP
    if dataset_provider is None:
        dataset_provider = get_dataset_provider(config.dataset_config)

    state = GlobalState()
    state.cfg = cfg
    # TODO: Freeze state.cfg

    setup_logging(
        logging_level=cfg.logger_config.logging_level,
        filter_warning=cfg.logger_config.filter_warnings,
        modules_to_filter=cfg.logger_config.modules_to_filter,
        set_level_for_all_loggers=cfg.logger_config.set_level_for_all_loggers,
    )

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(cfg=cfg)

    timers = state.timers

    if cfg.logger_config.log_progress:
        append_to_progress_log(cfg.checkpoint_config.save, "Starting job")

    if cfg.ft_config and cfg.ft_config.enable_ft_package:
        fault_tolerance.setup(cfg, state)
        fault_tolerance.maybe_setup_simulated_fault(cfg.ft_config)

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options(cfg.model_config, cfg.train_config.micro_batch_size)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    start_time_tensor = torch.tensor([state.start_time], dtype=torch.double, device="cuda")
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    state.start_time = start_time_tensor.item()

    print_rank_0("time to initialize megatron (seconds): {:.3f}".format(time.time() - state.start_time))
    barrier_and_log("after megatron is initialized")

    # Context used for persisting some state between checkpoint saves.
    checkpointing_context = _init_checkpointing_context(cfg.checkpoint_config)

    # Tokenizer
    timers("tokenizer-setup", log_level=0).start(barrier=True)
    tokenizer = build_tokenizer(
        cfg.tokenizer_config,
        make_vocab_size_divisible_by=cfg.model_config.make_vocab_size_divisible_by,
        tensor_model_parallel_size=cfg.model_config.tensor_model_parallel_size,
    )
    if not cfg.model_config.vocab_size:
        cfg.model_config.vocab_size = tokenizer.vocab_size

    cfg.dataset_config.tokenizer = tokenizer
    timers("tokenizer-setup").stop()
    barrier_and_log("after tokenizer is built")

    # Load base model from pretrained checkpoint
    model = get_base_model(cfg.model_config)
    timers("load-base-checkpoint", log_level=0).start(barrier=True)
    load_checkpoint(
        state,
        model,
        optimizer=None,
        scheduler=None,
    )
    timers("load-base-checkpoint").stop(barrier=True)
    timers.log(["load-base-checkpoint"])
    # TODO: Conditionally handle resuming from checkpoint saved during training
    # meaning only the adapter weights should be loaded

    # Model, optimizer, and learning rate.
    timers("model-and-optimizer-setup", log_level=0).start(barrier=True)
    model = peft(model)
    model = get_distributed_model(
        model,
        cfg.ddp_config,
        use_torch_fsdp2=cfg.dist_config.use_torch_fsdp2,
        overlap_param_gather_with_optimizer_step=cfg.optimizer_config.overlap_param_gather_with_optimizer_step,
        data_parallel_random_init=cfg.rng_config.data_parallel_random_init,
    )
    cfg.model_config.timers = timers
    cfg.optimizer_config.timers = timers
    optimizer, scheduler = setup_optimizer(
        optimizer_config=cfg.optimizer_config,
        scheduler_config=cfg.scheduler_config,
        model=model,
        use_gloo_process_groups=cfg.dist_config.use_gloo_process_groups,
    )
    _update_model_config_funcs(
        model,
        cfg.model_config,
        cfg.ddp_config,
        optimizer,
        align_grad_reduce=cfg.dist_config.align_grad_reduce,
    )
    timers("model-and-optimizer-setup").stop()
    barrier_and_log("after model, optimizer, and learning rate scheduler are built")

    # TODO: Conditionally handle resuming additional states from checkpoint saved during training

    # Data stuff.
    timers("train/valid/test-data-iterators-setup", log_level=0).start(barrier=True)
    if "tokenizer" in inspect.signature(train_valid_test_datasets_provider).parameters:
        train_valid_test_datasets_provider = partial(train_valid_test_datasets_provider, tokenizer=tokenizer)

    train_data_iterator, valid_data_iterator, test_data_iterator = setup_data_iterators(
        cfg=cfg,
        train_state=state.train_state,
        model_length=len(model),
        train_valid_test_datasets_provider=train_valid_test_datasets_provider,
    )
    timers("train/valid/test-data-iterators-setup").stop()
    barrier_and_log("after dataloaders are built")

    # if args.enable_ft_package and ft_integration.get_rank_monitor_client() is not None:
    #     ft_integration.get_rank_monitor_client().init_workload_monitoring()
    #     ft_timeouts = ft_integration.get_rank_monitor_client().timeouts
    #     print_rank_0(f"Fault tolerance client initialized. Timeouts: {ft_timeouts}")

    timers.log(["model-and-optimizer-setup", "train/valid/test-data-iterators-setup"], barrier=True)
    # TRAINING
    if not config.train_config.skip_train:
        print_rank_0("training ...")
        if state.train_state.do_train and config.train_config.train_iters > 0:
            train(
                forward_step_func,
                model,
                optimizer,
                scheduler,
                train_data_iterator,
                valid_data_iterator,
                state,
                ckpt_context,
                peft=peft
            )

        barrier_and_log("after training is done")
        ckpt_config = config.checkpoint_config
        if ckpt_config.save and state.train_state.step != 0 and ckpt_config.save_interval != 0:
            save_checkpoint(
                state,
                model,
                optimizer,
                scheduler,
                state.train_state.floating_point_operations_so_far,
                ckpt_context,
                train_data_iterator=train_data_iterator,
                peft=peft,
            )

    else:
        print_rank_0("skipping training ...")

    iteration = state.train_state.step

    # VALIDATION
    if state.train_state.do_valid:
        prefix = f"iteration {iteration} on validation set"
        evaluate_and_print_results(
            state,
            prefix,
            forward_step_func,
            valid_data_iterator,
            model,
            config.model_config,
            verbose=True,
            write_to_tensorboard=not config.train_config.skip_train,
        )
    if state.train_state.do_test:
        prefix = f"iteration {iteration} on test set"
        evaluate_and_print_results(
            state,
            prefix,
            forward_step_func,
            test_data_iterator,
            model,
            config.model_config,
            verbose=True,
            write_to_tensorboard=not config.train_config.skip_train,
        )

    _finish_train(state)
