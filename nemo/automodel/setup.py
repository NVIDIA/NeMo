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
from typing import Iterator, NamedTuple, Optional

import torch
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from nemo.automodel.checkpointing import load_checkpoint
from nemo.automodel.config import ConfigContainer
from nemo.automodel.data import build_train_valid_test_data_iterators
from nemo.automodel.init import initialize_automodel
from nemo.automodel.model import get_model_from_config
from nemo.automodel.optim import setup_optimizer
from nemo.tron import fault_tolerance
from nemo.tron.checkpointing import checkpoint_exists
from nemo.tron.state import GlobalState
from nemo.tron.tokenizers.tokenizer import build_tokenizer
from nemo.tron.utils.common_utils import (append_to_progress_log,
                                          barrier_and_log, get_rank_safe,
                                          get_world_size_safe, print_rank_0)
from nemo.tron.utils.log_utils import setup_logging


class SetupOutput(NamedTuple):
    state: GlobalState
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: OptimizerParamScheduler
    train_data_iterator: Optional[Iterator | list[Iterator]]
    valid_data_iterator: Optional[Iterator | list[Iterator]]
    test_data_iterator: Optional[Iterator | list[Iterator]]


def setup(
    cfg: ConfigContainer,
    train_valid_test_datasets_provider,
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

    # if cfg.ft_config and cfg.ft_config.enable_ft_package:
    #     fault_tolerance.setup(cfg, state)
    #     fault_tolerance.maybe_setup_simulated_fault(cfg.ft_config)

    # TODO: Set pytorch JIT layer fusion options and warmup JIT functions for automodel if needed.
    # set_jit_fusion_options(cfg.model_config, cfg.train_config.micro_batch_size)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    start_time_tensor = torch.tensor([state.start_time], dtype=torch.double, device="cuda")
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    state.start_time = start_time_tensor.item()

    print_rank_0("time to initialize automodel (seconds): {:.3f}".format(time.time() - state.start_time))
    barrier_and_log("after automodel is initialized")

    # Tokenizer
    timers("tokenizer-setup", log_level=0).start(barrier=True)
    tokenizer = build_tokenizer(
        cfg.tokenizer_config,
        make_vocab_size_divisible_by=cfg.model_config.make_vocab_size_divisible_by,
        tensor_model_parallel_size=1,  # TODO: Change when TP support is added
    )

    cfg.dataset_config.tokenizer = tokenizer
    timers("tokenizer-setup").stop()
    barrier_and_log("after tokenizer is built")

    # Model, optimizer, and learning rate.
    timers("model-and-optimizer-setup", log_level=0).start(barrier=True)
    model = get_model_from_config(
        cfg.model_config,
        use_torch_fsdp2=cfg.dist_config.use_torch_fsdp2,
        wrap_with_ddp=get_world_size_safe() > 1 and not cfg.dist_config.use_torch_fsdp2,
        ddp_kwargs=cfg.model_config.ddp_kwargs,
    )
    optimizer, scheduler = setup_optimizer(
        optimizer_config=cfg.optimizer_config,
        scheduler_config=cfg.scheduler_config,
        model=model,
    )
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
    if "tokenizer" in inspect.signature(train_valid_test_datasets_provider).parameters:
        train_valid_test_datasets_provider = partial(train_valid_test_datasets_provider, tokenizer=tokenizer)

    train_data_iterator, valid_data_iterator, test_data_iterator = build_train_valid_test_data_iterators(
        cfg=cfg,
        train_state=state.train_state,
        data_parallel_rank=get_rank_safe(),  # TODO: Change when Model Parallel support is added
        data_parallel_size=get_world_size_safe(),  # TODO: Change when Model Parallel support is added
        build_train_valid_test_datasets_provider=train_valid_test_datasets_provider,
    )
    timers("train/valid/test-data-iterators-setup").stop()
    barrier_and_log("after dataloaders are built")

    # if args.enable_ft_package and ft_integration.get_rank_monitor_client() is not None:
    #     ft_integration.get_rank_monitor_client().init_workload_monitoring()
    #     ft_timeouts = ft_integration.get_rank_monitor_client().timeouts
    #     print_rank_0(f"Fault tolerance client initialized. Timeouts: {ft_timeouts}")

    # Print setup timing.
    print_rank_0("done with setup ...")
    timers.log(["model-and-optimizer-setup", "train/valid/test-data-iterators-setup"], barrier=True)

    return SetupOutput(
        state,
        model,
        optimizer,
        scheduler,
        train_data_iterator,
        valid_data_iterator,
        test_data_iterator,
    )
