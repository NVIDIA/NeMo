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
from typing import Any, Dict, NamedTuple, Optional

import torch
from megatron.core.distributed import (DistributedDataParallel,
                                       DistributedDataParallelConfig,
                                       finalize_model_grads)
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.rerun_state_machine import RerunDataIterator
from megatron.core.transformer import MegatronModule

from nemo.collections.llm.gpt.model.base import GPTConfig
from nemo.collections.llm.t5.model.t5 import T5Config
from nemo.tron import fault_tolerance
from nemo.tron.checkpointing import checkpoint_exists, load_checkpoint
from nemo.tron.config import CheckpointConfig, ConfigContainer
from nemo.tron.data.loaders import setup_data_iterators
from nemo.tron.init import initialize_megatron, set_jit_fusion_options
from nemo.tron.model import get_model_from_config
from nemo.tron.optim import setup_optimizer
from nemo.tron.state import GlobalState
from nemo.tron.tokenizers.tokenizer import build_tokenizer
from nemo.tron.utils.common_utils import (append_to_progress_log,
                                          barrier_and_log, print_rank_0)
from nemo.tron.utils.log_utils import setup_logging
from nemo.utils.import_utils import safe_import

_, HAVE_RESIL = safe_import("nvidia_resiliency_ext.checkpointing")

try:
    from megatron.core.distributed import \
        TorchFullyShardedDataParallel  # noqa: F401

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False


class SetupOutput(NamedTuple):
    state: GlobalState
    model: MegatronModule
    optimizer: MegatronOptimizer
    scheduler: OptimizerParamScheduler
    train_data_iterator: Optional[RerunDataIterator | list[RerunDataIterator]]
    valid_data_iterator: Optional[RerunDataIterator | list[RerunDataIterator]]
    test_data_iterator: Optional[RerunDataIterator | list[RerunDataIterator]]
    checkpointing_context: dict[str, Any]


def setup(
    cfg: ConfigContainer,
    train_valid_test_datasets_provider,
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
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

    initialize_megatron(
        cfg=cfg,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
    )

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

    # Model, optimizer, and learning rate.
    timers("model-and-optimizer-setup", log_level=0).start(barrier=True)
    model = get_model_from_config(
        cfg.model_config,
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
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2 and cfg.dist_config.use_torch_fsdp2,
        )
        timers("load-checkpoint").stop(barrier=True)
        timers.log(["load-checkpoint"])

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
        checkpointing_context,
    )


def _init_checkpointing_context(checkpoint_config: CheckpointConfig) -> Dict[str, Any]:
    # Context used for persisting some state between checkpoint saves.
    if checkpoint_config.non_persistent_ckpt_type != "local":
        return {}

    if not HAVE_RESIL:
        raise RuntimeError(
            "The 'nvidia_resiliency_ext' module is required for local "
            "checkpointing but was not found. Please ensure it is installed."
        )

    from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import \
        LocalCheckpointManager
    from nvidia_resiliency_ext.checkpointing.local.replication.strategies import \
        CliqueReplicationStrategy

    if checkpoint_config.replication:
        repl_strategy = CliqueReplicationStrategy.from_replication_params(
            checkpoint_config.replication_jump,
            checkpoint_config.replication_factor,
        )
    else:
        repl_strategy = None

    checkpointing_context = {
        "local_checkpoint_manager": LocalCheckpointManager(
            checkpoint_config.non_persistent_local_ckpt_dir,
            repl_strategy=repl_strategy,
        )
    }
    return checkpointing_context


def _update_model_config_funcs(
    model: MegatronModule,
    model_config: GPTConfig | T5Config,
    ddp_config: DistributedDataParallelConfig,
    optimizer: MegatronOptimizer,
    *,
    align_grad_reduce: bool = True,
) -> None:
    """Update model config sync funcs based on initialized model."""
    if isinstance(model[0], DistributedDataParallel) and ddp_config.overlap_grad_reduce:
        assert model_config.no_sync_func is None, (
            "When overlap_grad_reduce is True, config.no_sync_func must be None; "
            "a custom no_sync_func is not supported when overlapping grad-reduce"
        )
    model_config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
    if len(model) == 1:
        model_config.no_sync_func = model_config.no_sync_func[0]
    if align_grad_reduce:
        model_config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
        if len(model) == 1:
            model_config.grad_sync_func = model_config.grad_sync_func[0]
    if ddp_config.overlap_param_gather and ddp_config.align_param_gather:
        model_config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            model_config.param_sync_func = model_config.param_sync_func[0]
    model_config.finalize_model_grads_func = finalize_model_grads
    model_config.grad_scale_func = optimizer.scale_loss
