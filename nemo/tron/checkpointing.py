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

"""Input/output checkpointing."""

import contextlib
import os
import random
import shutil
import sys
import threading
from enum import Enum, auto
from functools import lru_cache
from logging import getLogger
from pathlib import Path
from time import time
from typing import Optional

import numpy as np
import torch
import yaml
from megatron.core import dist_checkpointing, mpu, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.serialization import (
    get_default_load_sharded_strategy, get_default_save_sharded_strategy)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper, FullyParallelSaveStrategyWrapper)
from megatron.core.fp8_utils import is_float8tensor
from megatron.core.num_microbatches_calculator import update_num_microbatches
from megatron.core.rerun_state_machine import get_rerun_state_machine

from nemo.tron import fault_tolerance
from nemo.tron.config import ConfigContainer
# from . import ft_integration
from nemo.tron.state import GlobalState, TrainState
from nemo.tron.utils import wandb_utils
from nemo.tron.utils.async_utils import (is_empty_async_queue,
                                         schedule_async_save)
from nemo.tron.utils.common_utils import (append_to_progress_log,
                                          dump_dataclass_to_yaml,
                                          get_rank_safe, get_world_size_safe,
                                          is_last_rank, print_rank_0,
                                          unwrap_model, use_dist_ckpt)

# [ModelOpt]: Import
try:
    from modelopt.torch.opt.plugins import (restore_modelopt_state,
                                            restore_sharded_modelopt_state,
                                            save_modelopt_state,
                                            save_sharded_modelopt_state)

    has_nvidia_modelopt = True
except Exception:
    has_nvidia_modelopt = False

TRAIN_STATE_FILE = "train_state.pt"
TRACKER_PREFIX = "latest"
CONFIG_FILE = "run_config.yaml"
_CHECKPOINT_VERSION = None

logger = getLogger(__name__)
_NON_PERSISTENT_CKPT_SUBDIR = "non_persistent"


def set_checkpoint_version(value):
    global _CHECKPOINT_VERSION
    if _CHECKPOINT_VERSION is not None:
        assert _CHECKPOINT_VERSION == value, "checkpoint versions do not match"
    _CHECKPOINT_VERSION = value


def get_checkpoint_version():
    global _CHECKPOINT_VERSION
    return _CHECKPOINT_VERSION


# TODO: Convert this to ConfigContainer
# def check_checkpoint_args(checkpoint_args):
#     """Ensure fixed arguments for a model are the same for the input
#     arguments and the one retrieved from checkpoint."""
#     args = get_args()

#     def _compare(arg_name, old_arg_name=None, default=None):
#         if old_arg_name is not None:
#             ckpt_arg_name = old_arg_name
#         else:
#             ckpt_arg_name = arg_name
#         if default is not None:
#             checkpoint_value = getattr(checkpoint_args, ckpt_arg_name, default)
#         else:
#             checkpoint_value = getattr(checkpoint_args, ckpt_arg_name)
#         args_value = getattr(args, arg_name)
#         error_message = "{} value from checkpoint ({}) is not equal to the input argument value ({}).".format(
#             arg_name, checkpoint_value, args_value
#         )
#         assert checkpoint_value == args_value, error_message

#     _compare("num_layers")
#     _compare("hidden_size")
#     _compare("num_attention_heads")
#     _compare("add_position_embedding", default=True)
#     if args.vocab_file:
#         _compare("max_position_embeddings")
#         _compare("make_vocab_size_divisible_by")
#         if not args.use_dist_ckpt:
#             _compare("padded_vocab_size")
#         _compare("tokenizer_type")
#     if args.data_parallel_random_init:
#         _compare("data_parallel_random_init")
#     if get_checkpoint_version() < 3.0:
#         _compare("tensor_model_parallel_size", old_arg_name="model_parallel_size")
#     if get_checkpoint_version() >= 3.0 and not args.use_dist_ckpt:
#         _compare("tensor_model_parallel_size")
#         _compare("pipeline_model_parallel_size")


def ensure_directory_exists(filename, check_parent=True):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename) if check_parent else filename
    os.makedirs(dirname, exist_ok=True)


def get_checkpoint_name(
    checkpoints_path,
    iteration,
    release=False,
    pipeline_parallel=None,
    tensor_rank=None,
    pipeline_rank=None,
    expert_parallel=None,
    expert_rank=None,
    return_base_dir=False,
    basename="model_optim_rng.pt",
):
    """Determine the directory name for this rank's checkpoint."""
    if release:
        directory = "release"
    else:
        directory = "iter_{:07d}".format(iteration)
    if return_base_dir:
        common_path = os.path.join(checkpoints_path, directory)
        return common_path

    # Use both the tensor and pipeline MP rank.
    if pipeline_parallel is None:
        pipeline_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
    if tensor_rank is None:
        tensor_rank = mpu.get_tensor_model_parallel_rank()
    if pipeline_rank is None:
        pipeline_rank = mpu.get_pipeline_model_parallel_rank()
    if expert_parallel is None:
        expert_parallel = mpu.get_expert_model_parallel_world_size() > 1
    if expert_rank is None:
        expert_rank = mpu.get_expert_model_parallel_rank()

    # Use both the tensor and pipeline MP rank. If using the distributed
    # optimizer, then the optimizer's path must additionally include the
    # data parallel rank.
    if not pipeline_parallel:
        common_path = os.path.join(checkpoints_path, directory, f"mp_rank_{tensor_rank:02d}")
    else:
        common_path = os.path.join(checkpoints_path, directory, f"mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}")

    if expert_parallel:
        common_path = common_path + f"_{expert_rank:03d}"

    return os.path.join(common_path, basename)


def get_distributed_optimizer_checkpoint_name(model_checkpoint_name):
    return os.path.join(os.path.dirname(model_checkpoint_name), "distrib_optim.pt")


def find_checkpoint_rank_0(checkpoints_path, iteration, release=False):
    """Finds the checkpoint for rank 0 without knowing if we are using
    pipeline parallelism/expert parallelism or not.

    Since the checkpoint naming scheme changes if pipeline or expert
    parallelism is present, we need to look for both naming schemes if
    we don't know if the checkpoint has pipeline or expert parallelism.
    """

    # Look for checkpoint with no pipelining and no expert parallelism
    filename = get_checkpoint_name(
        checkpoints_path,
        iteration,
        release,
        pipeline_parallel=False,
        tensor_rank=0,
        pipeline_rank=0,
        expert_parallel=False,
        expert_rank=0,
    )
    if os.path.isfile(filename):
        return filename

    # Look for checkpoint with no pipelining and expert parallelism
    filename = get_checkpoint_name(
        checkpoints_path,
        iteration,
        release,
        pipeline_parallel=False,
        tensor_rank=0,
        pipeline_rank=0,
        expert_parallel=True,
        expert_rank=0,
    )
    if os.path.isfile(filename):
        return filename

    # Look for checkpoint with pipelining and no expert parallelism
    filename = get_checkpoint_name(
        checkpoints_path,
        iteration,
        release,
        pipeline_parallel=True,
        tensor_rank=0,
        pipeline_rank=0,
        expert_parallel=False,
        expert_rank=0,
    )
    if os.path.isfile(filename):
        return filename

    # Look for checkpoint with pipelining and expert parallelism
    filename = get_checkpoint_name(
        checkpoints_path,
        iteration,
        release,
        pipeline_parallel=True,
        tensor_rank=0,
        pipeline_rank=0,
        expert_parallel=True,
        expert_rank=0,
    )
    if os.path.isfile(filename):
        return filename

    # Look for a distributed checkpoint
    filename = get_checkpoint_name(checkpoints_path, iteration, release, pipeline_parallel=True, return_base_dir=True)
    if dist_checkpointing.check_is_distributed_checkpoint(filename):
        return filename

    return None


def get_checkpoint_train_state_filename(checkpoints_path, prefix: Optional[str] = None):
    """Tracker file records the latest train state to restart from."""
    if prefix is None:
        return os.path.join(checkpoints_path, TRAIN_STATE_FILE)
    else:
        return os.path.join(checkpoints_path, f"{prefix}_{TRAIN_STATE_FILE}")


def get_checkpoint_run_config_filename(checkpoints_path):
    """Checkpoint config file records the run config."""
    return os.path.join(checkpoints_path, CONFIG_FILE)


def checkpoint_exists(checkpoints_path):
    if checkpoints_path is None:
        return False
    return os.path.exists(os.path.join(checkpoints_path, f"{TRACKER_PREFIX}_{TRAIN_STATE_FILE}"))


@lru_cache()
def read_train_state(train_state_filename: str):
    """
    Read the TrainState saved via TrainState.state_dict. On rank 0 load from the file,
    then broadcast the TrainState object (or an error flag) to all processes.
    If an error occurs on rank 0, all ranks will raise an exception.
    """
    state_obj = [None]

    if get_rank_safe() == 0:
        try:
            state_dict = torch.load(train_state_filename, map_location="cpu")
            ts = TrainState()
            ts.load_state_dict(state_dict)
            state_obj[0] = ts
        except Exception as e:
            error_msg = f"ERROR: Unable to load train state file {train_state_filename}: {e}"
            sys.stderr.write(error_msg + "\n")
            state_obj[0] = {"error": True, "msg": error_msg}

    if torch.distributed.is_initialized():
        print_rank_0(f"Broadcasting TrainState from rank 0 to all {get_world_size_safe()} ranks")
        torch.distributed.broadcast_object_list(state_obj, src=0)

    if isinstance(state_obj[0], dict) and state_obj[0].get("error", False):
        raise RuntimeError(state_obj[0]["msg"])

    return state_obj[0]


@lru_cache()
def read_run_config(run_config_filename: str):
    """
    Read the run config saved as YAML. On rank 0 load from the file,
    then broadcast the config object (or an error flag) to all processes.
    If an error occurs on rank 0, all ranks will raise an exception.
    """
    config_obj = [None]

    if get_rank_safe() == 0:
        try:
            with open(run_config_filename, "r") as f:
                config_dict = yaml.safe_load(f)
            config_obj[0] = config_dict
        except Exception as e:
            error_msg = f"ERROR: Unable to load config file {run_config_filename}: {e}"
            sys.stderr.write(error_msg + "\n")
            config_obj[0] = {"error": True, "msg": error_msg}

    if torch.distributed.is_initialized():
        print_rank_0(f"Broadcasting config from rank 0 to all {get_world_size_safe()} ranks")
        torch.distributed.broadcast_object_list(config_obj, src=0)

    if isinstance(config_obj[0], dict) and config_obj[0].get("error", False):
        raise RuntimeError(config_obj[0]["msg"])

    return config_obj[0]


def get_rng_state(data_parallel_random_init: bool, use_dist_ckpt: bool = False):
    """collect rng state across data parallel ranks"""
    rng_state = {
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states(),
    }

    rng_state_list = None
    if torch.distributed.is_initialized() and mpu.get_data_parallel_world_size() > 1 and data_parallel_random_init:
        rng_state_list = [None for i in range(mpu.get_data_parallel_world_size())]
        torch.distributed.all_gather_object(rng_state_list, rng_state, group=mpu.get_data_parallel_group())
    else:
        rng_state_list = [rng_state]

    if use_dist_ckpt:
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        rng_state_list = ShardedObject(
            "rng_state",
            rng_state_list,
            (pp_size, tp_size),
            (pp_rank, tp_rank),
            replica_id=mpu.get_data_parallel_rank(with_context_parallel=True),
        )

    return rng_state_list


class CheckpointType(Enum):
    LEGACY = auto()
    LOCAL = auto()
    GLOBAL = auto()


def save_checkpoint(
    state: GlobalState,
    model,
    optimizer,
    opt_param_scheduler,
    num_floating_point_operations_so_far,
    checkpointing_context=None,
    pipeline_rank=None,
    expert_rank=None,
    tensor_rank=None,
    pipeline_parallel=None,
    expert_parallel=None,
    non_persistent_ckpt=False,
    train_data_iterator=None,
    preprocess_common_state_dict_fn=None,
):
    """Save a model, optimizer and optionally dataloader checkpoint.

    Checkpointing context is used to persist some checkpointing state
    throughout a single job. Must be initialized externally (not used if None).

    If non_persistent_ckpt is True,
    the checkpoint will be saved with special functionality for removing old checkpoints.
    There are several types of non-persistent checkpoints:
    "global" - Saved as a standard checkpoint (e.g., on Lustre) with old checkpoints being removed.
    "local" - Each rank saves a portion of the checkpoint locally (e.g., on SSD/ramdisk).

    Dataloader checkpoint is only saved if the dataloader supports it. Currently this applies only
    to the Megatron Energon dataloader (multimodal) and not the built-in Megatron dataloader (text-only).
    """
    train_state = state.train_state
    start_ckpt = time()
    cfg = state.cfg
    ckpt_cfg = cfg.checkpoint_config

    if ckpt_cfg.async_save and not is_empty_async_queue():
        print_rank_0(
            "WARNING: Starting a checkpoint save before previous has finished. Consider increasing the checkpoint interval."
        )

    # Monitor for the checkpointing timeout (no-op if FT is not enabled)
    fault_tolerance.on_checkpointing_start(state)

    # Only rank zero of the data parallel writes to the disk.
    model = unwrap_model(model)

    # Handle non_persistent_ckpt flag.
    ckpt_type = CheckpointType.GLOBAL if use_dist_ckpt(ckpt_cfg.ckpt_format) else CheckpointType.LEGACY
    save_dir = ckpt_cfg.save
    if non_persistent_ckpt:
        if ckpt_cfg.non_persistent_ckpt_type == "global":
            ckpt_type = CheckpointType.GLOBAL
            save_dir = (
                ckpt_cfg.non_persistent_global_ckpt_dir
                if ckpt_cfg.non_persistent_global_ckpt_dir
                else os.path.join(save_dir, _NON_PERSISTENT_CKPT_SUBDIR)
            )
            # TODO Can we ensure the previous checkpoint is saved? We don't want to allow two saves in parallel.
            cleanup_old_non_persistent_checkpoint(save_dir, leave_ckpt_num=1, do_async=ckpt_cfg.async_save)
        elif ckpt_cfg.non_persistent_ckpt_type == "local":
            ckpt_type = CheckpointType.LOCAL
            save_dir = checkpointing_context["local_checkpoint_manager"].local_ckpt_dir
        else:
            assert (
                False
            ), f"Please use local or global non-persistent checkpoints(got: {ckpt_cfg.non_persistent_ckpt_type})"

    ckpt_format = ckpt_cfg.ckpt_format if ckpt_type == CheckpointType.GLOBAL else "torch"
    print_rank_0(f"saving checkpoint at iteration {train_state.step:7d} to {save_dir} in {ckpt_format} format")

    # Collect rng state across data parallel ranks.
    rng_state = get_rng_state(
        data_parallel_random_init=cfg.rng_config.data_parallel_random_init,
        use_dist_ckpt=ckpt_type != CheckpointType.LEGACY,
    )

    # Collect rerun state across all ranks
    rerun_state_machine = get_rerun_state_machine()
    rerun_state = rerun_state_machine.state_dict(
        data_iterator=train_data_iterator,
        ckpt_format=ckpt_cfg.ckpt_format,
    )

    # Checkpoint name.
    return_base_dir = ckpt_type != CheckpointType.LEGACY
    checkpoint_name = get_checkpoint_name(
        save_dir,
        train_state.step,
        release=False,
        pipeline_parallel=pipeline_parallel,
        tensor_rank=tensor_rank,
        pipeline_rank=pipeline_rank,
        expert_parallel=expert_parallel,
        expert_rank=expert_rank,
        return_base_dir=return_base_dir,
    )

    # Save dataloader state if the dataloader supports it (currently only Megatron Energon).
    maybe_save_dataloader_state(
        train_data_iterator, train_state.step, getattr(cfg.dataset_config, "dataloader_save", None)
    )

    # Save distributed optimizer's custom parameter state.
    if (
        cfg.optimizer_config.use_distributed_optimizer
        and ckpt_cfg.save_optim
        and optimizer is not None
        and ckpt_type == CheckpointType.LEGACY
    ):
        optim_checkpoint_name = get_distributed_optimizer_checkpoint_name(checkpoint_name)
        ensure_directory_exists(optim_checkpoint_name)
        if not getattr(optimizer, "is_stub_optimizer", False):
            optimizer.save_parameter_state(optim_checkpoint_name)

    async_save_request = None
    if ckpt_cfg.async_save:
        if ckpt_type == CheckpointType.LEGACY:
            raise NotImplementedError("Async checkpoint save not implemented for legacy checkpoints")
        elif ckpt_type == CheckpointType.GLOBAL and ckpt_cfg.ckpt_format != "torch_dist":
            raise NotImplementedError(
                f"Async checkpoint save not implemented for {ckpt_cfg.ckpt_format} distributed checkpoint format"
            )

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    # Collect cfg, model, RNG.
    if (
        not torch.distributed.is_initialized()
        or mpu.get_expert_data_parallel_rank() == 0
        or ckpt_type != CheckpointType.LEGACY
    ):
        optim_sd_kwargs = {}
        if ckpt_type != CheckpointType.LEGACY and cfg.optimizer_config.use_distributed_optimizer:
            optim_sd_kwargs["sharding_type"] = (
                "fully_sharded_model_space" if ckpt_cfg.fully_parallel_save else "dp_zero_gather_scatter"
            )
            print_rank_0(f"Storing distributed optimizer sharded state of type {optim_sd_kwargs['sharding_type']}")
        state_dict = generate_state_dict(
            cfg,
            model,
            optimizer,
            opt_param_scheduler,
            rng_state,
            use_dist_ckpt=ckpt_type != CheckpointType.LEGACY,
            iteration=train_state.step,
            optim_sd_kwargs=optim_sd_kwargs,
            rerun_state=rerun_state,
        )

        if ckpt_type == CheckpointType.GLOBAL:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                # TODO Handle non-empty directories (e.g., after a crash during saving).
                ensure_directory_exists(checkpoint_name, check_parent=False)
            if checkpointing_context is not None and "save_strategy" in checkpointing_context:
                save_strategy = checkpointing_context["save_strategy"]
                # Already saved once before - don't need to rerun sharding validation
                validate_sharding_integrity = not ckpt_cfg.ckpt_assume_constant_structure
            else:
                validate_sharding_integrity = True
                save_strategy = get_default_save_sharded_strategy(ckpt_cfg.ckpt_format)
                if ckpt_cfg.ckpt_assume_constant_structure and ckpt_cfg.ckpt_format == "torch_dist":
                    save_strategy.use_cached_ckpt_structure = ckpt_cfg.ckpt_assume_constant_structure
                    if checkpointing_context is not None and "load_strategy" in checkpointing_context:
                        cached_global_metadata = getattr(
                            checkpointing_context["load_strategy"], "cached_global_metadata", None
                        )
                        if cached_global_metadata is not None:
                            logger.debug("Plugging in the read metadata from the load strategy...")
                            save_strategy.cached_global_metadata = cached_global_metadata
                        else:
                            logger.debug("Failed to plug in the read metadata from the load strategy...")

                if ckpt_cfg.fully_parallel_save:
                    save_strategy = FullyParallelSaveStrategyWrapper(
                        save_strategy,
                        mpu.get_data_parallel_group(with_context_parallel=True),
                        ckpt_cfg.ckpt_assume_constant_structure,
                    )
            # Store save strategy for future checkpoint saves
            if checkpointing_context is not None:
                checkpointing_context["save_strategy"] = save_strategy
            end_ckpt = time()
            logger.debug(f"rank: {rank}, takes {end_ckpt - start_ckpt} to prepare state dict for ckpt ")
            async_save_request = dist_checkpointing.save(
                state_dict,
                checkpoint_name,
                save_strategy,
                async_sharded_save=ckpt_cfg.async_save,
                validate_access_integrity=validate_sharding_integrity,
                preprocess_common_before_consistancy_check=preprocess_common_state_dict_fn,
            )
            # [ModelOpt]: save sharded modelopt_state
            if has_nvidia_modelopt:
                save_sharded_modelopt_state(model, checkpoint_name, (ckpt_cfg.ckpt_format, 1))
        else:
            # [ModelOpt]: Inject modelopt_state into state_dict
            if has_nvidia_modelopt:
                if ckpt_type == CheckpointType.LOCAL:
                    print_rank_0("WARNING: Local checkpointing does not support nvidia_modelopt.")
                else:
                    save_modelopt_state(model, state_dict)

            end_ckpt = time()
            logger.debug(f"rank: {rank}, takes {end_ckpt - start_ckpt} to prepare state dict for ckpt ")
            if ckpt_type == CheckpointType.LOCAL:
                try:
                    from megatron.core.dist_checkpointing.tensor_aware_state_dict import \
                        MCoreTensorAwareStateDict
                except ModuleNotFoundError:
                    raise RuntimeError(
                        "The 'nvidia_resiliency_ext' module is required for local "
                        "checkpointing but was not found. Please ensure it is installed."
                    )

                algo = ckpt_cfg.non_persistent_local_ckpt_algo
                cached_metadata = None
                if ckpt_cfg.ckpt_assume_constant_structure and "local_checkpoint_cache" in checkpointing_context:
                    cached_metadata = checkpointing_context["local_checkpoint_cache"]
                state_dict_for_save, cacheable_metadata = MCoreTensorAwareStateDict.from_state_dict(
                    state_dict,
                    algo=algo,
                    cached_metadata=cached_metadata,
                    parallelization_group=mpu.get_data_parallel_group(with_context_parallel=True),
                )
                async_save_request = checkpointing_context["local_checkpoint_manager"].save(
                    state_dict_for_save, train_state.step, is_async=bool(ckpt_cfg.async_save)
                )
                checkpointing_context["local_checkpoint_cache"] = cacheable_metadata
            else:
                assert ckpt_type == CheckpointType.LEGACY
                # Save.
                ensure_directory_exists(checkpoint_name)
                torch.save(state_dict, checkpoint_name)
    start_misc = time()
    if ckpt_type != CheckpointType.LOCAL:
        if not ckpt_cfg.async_save:
            assert async_save_request is None
            # Wait so everyone is done (necessary)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    # And update the latest train state
    if get_rank_safe() == 0:
        train_state_local_filename = get_checkpoint_train_state_filename(checkpoint_name)
        train_state_global_filename = get_checkpoint_train_state_filename(save_dir, prefix=TRACKER_PREFIX)
        config_filename = get_checkpoint_run_config_filename(checkpoint_name)
        if ckpt_type == CheckpointType.LOCAL:

            def train_state_finalize_fn():
                print_rank_0(f"  successfully saved local checkpoint from iteration {train_state.step:7d}")
                if cfg.logger_config.log_progress and ckpt_cfg.async_save:
                    append_to_progress_log(
                        ckpt_cfg.save, f"Saved async local checkpoint\tIteration: {train_state.step}", barrier=False
                    )

        else:
            train_state_dict = train_state.state_dict()

            def train_state_finalize_fn():
                train_state_dict["floating_point_operations_so_far"] = torch.tensor(
                    num_floating_point_operations_so_far, dtype=torch.float32
                )
                torch.save(train_state_dict, train_state_local_filename)
                shutil.copy(train_state_local_filename, train_state_global_filename)

                cfg.to_yaml(config_filename)
                print_rank_0(
                    f"  successfully saved checkpoint from iteration {train_state_dict['step'].item():7d} to {ckpt_cfg.save} "
                    f"[ t {(tensor_rank if tensor_rank is not None else mpu.get_tensor_model_parallel_rank()) + 1}/{mpu.get_tensor_model_parallel_world_size()}, "
                    f"p {(pipeline_rank if pipeline_rank is not None else mpu.get_pipeline_model_parallel_rank()) + 1}/{mpu.get_pipeline_model_parallel_world_size()} ]"
                )
                if cfg.logger_config.log_progress and ckpt_cfg.async_save:
                    append_to_progress_log(
                        ckpt_cfg.save,
                        f"Saved async checkpoint\tIteration: {train_state_dict['step'].item()}",
                        barrier=False,
                    )

        if ckpt_cfg.async_save:
            assert async_save_request is not None
            async_save_request.add_finalize_fn(train_state_finalize_fn)
        else:
            train_state_finalize_fn()

    # Additional callback for wandb (last rank)
    if not torch.distributed.is_initialized() or is_last_rank():

        def wandb_finalize_fn():
            wandb_utils.on_save_checkpoint_success(
                checkpoint_name,
                save_dir,
                train_state.step,
                wandb_writer=state.wandb_logger,
            )

        if ckpt_cfg.async_save:
            assert async_save_request is not None
            async_save_request.add_finalize_fn(wandb_finalize_fn)
        else:
            wandb_finalize_fn()

    if ckpt_cfg.async_save:
        schedule_async_save(async_save_request)
        print_rank_0(f"  scheduled an async checkpoint save at iteration {train_state.step:7d} to {save_dir}")

    # Wait so everyone is done (not necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    end_misc = time()
    logger.debug(f"rank: {rank}, takes {end_misc - start_misc} to finalize ckpt save ")

    fault_tolerance.on_checkpointing_end(global_state=state, is_async_finalization=False)


def cleanup_old_non_persistent_checkpoint(save_dir, leave_ckpt_num=1, do_async=False):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    save_dir = Path(save_dir)

    iter_prefix = "iter_"
    iter_ckpts = save_dir.rglob(f"{iter_prefix}*")
    sorted_iter_ckpts = sorted(iter_ckpts, key=lambda ckpt_name: int(ckpt_name.name[len(iter_prefix) :]))
    if not sorted_iter_ckpts:
        return
    rm_iter_ckpts = sorted_iter_ckpts[:-leave_ckpt_num]
    print_rank_0(f"Non-persistent checkpoints scheduled for removal: {rm_iter_ckpts}")
    print_rank_0(f"Non-persistent checkpoints to be kept: {sorted_iter_ckpts[-leave_ckpt_num:]}")

    def remove_iter_ckpts(_iter_ckpts):
        for ckpt in _iter_ckpts:
            shutil.rmtree(ckpt)

    if do_async:
        threading.Thread(target=remove_iter_ckpts, args=(rm_iter_ckpts,)).start()
    else:
        remove_iter_ckpts(rm_iter_ckpts)


def maybe_save_dataloader_state(train_iterator, iteration, dataloader_save_path):
    """Saves dataloader state if the dataloader supports it.

    Currently, this is only used by Megatron Energon dataloader (multimodal) to store its state at a
    specific iteration. The Megatron built-in dataloader (text-only) creates index files upfront
    to track its state.

    If the provided dataloader has `save_state` method, then it is called to save the state.
    Otherwise, no state is saved.

    Args:
        train_iterator (iterable): Train dataloader.
        iteration (int): Current iteration.
        dataloader_save_path (str): Path where the dataloader state is saved.
    """
    # If no dataloader or saving path is provided, exit early, otherwise, raise an error.
    if train_iterator is None or dataloader_save_path is None or dataloader_save_path == "":
        return

    # If dataloader doesn't support saving state, raise an error.
    if not hasattr(train_iterator.iterable, "save_state"):
        raise RuntimeError(f"Could not find a save_state for the train_iterator of type {type(train_iterator)}")

    # Save dataloader state for each data parallel rank only once.
    first_rank = mpu.is_pipeline_first_stage(ignore_virtual=True) and mpu.get_tensor_model_parallel_rank() == 0
    if not first_rank:
        return

    dp_rank = mpu.get_data_parallel_rank()
    print(f"saving dataloader checkpoint at iteration {iteration} to {dataloader_save_path}")
    train_dataloader_state_dict = train_iterator.iterable.save_state()
    data_state_save_path = get_checkpoint_name(
        dataloader_save_path, iteration, basename=f"train_dataloader_dprank{dp_rank:03d}.pt"
    )

    torch.distributed.barrier(group=mpu.get_data_parallel_group())

    if mpu.get_data_parallel_rank() == 0:
        ensure_directory_exists(data_state_save_path)

    torch.distributed.barrier(group=mpu.get_data_parallel_group())

    dataloader_save_dict = {}
    dataloader_save_dict["dataloader_state_dict"] = train_dataloader_state_dict
    torch.save(dataloader_save_dict, data_state_save_path)


def generate_state_dict(
    cfg: ConfigContainer,
    model,
    optimizer,
    opt_param_scheduler,
    rng_state,
    use_dist_ckpt=False,
    iteration=None,
    optim_sd_kwargs=None,
    rerun_state=None,
):
    # Arguments, iteration, and model.
    state_dict = {}
    state_dict["checkpoint_version"] = 3.0
    if iteration is not None:
        state_dict["iteration"] = iteration

    if len(model) == 1:
        state_dict["model"] = (
            model[0].sharded_state_dict() if use_dist_ckpt else model[0].state_dict_for_save_checkpoint()
        )
    else:
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            state_dict["model%d" % i] = (
                model[i].sharded_state_dict() if use_dist_ckpt else model[i].state_dict_for_save_checkpoint()
            )
    # Optimizer stuff.
    if cfg.checkpoint_config.save_optim:
        if optimizer is not None and not getattr(optimizer, "is_stub_optimizer", False):
            state_dict["optimizer"] = (
                optimizer.sharded_state_dict(state_dict, **(optim_sd_kwargs or {}))
                if use_dist_ckpt
                else optimizer.state_dict()
            )
        if opt_param_scheduler is not None:
            state_dict["opt_param_scheduler"] = opt_param_scheduler.state_dict()

    # Rerun state
    state_dict["rerun_state_machine"] = rerun_state

    # RNG states.
    if cfg.checkpoint_config.save_rng:
        state_dict["rng_state"] = rng_state
    return state_dict


def _transpose_first_dim(t, num_splits, num_splits_first, model):
    input_shape = t.size()
    # We use a self_attention module but the values extracted aren't
    # specific to self attention so should work for cross attention as well
    while hasattr(model, "module"):
        model = model.module
    attention_module = model.language_model.encoder.layers[0].self_attention
    hidden_size_per_attention_head = attention_module.hidden_size_per_attention_head
    num_attention_heads_per_partition = attention_module.num_attention_heads_per_partition
    if num_splits_first:
        """[num_splits * np * hn, h]
        -->(view) [num_splits, np, hn, h]
        -->(tranpose) [np, num_splits, hn, h]
        -->(view) [np * num_splits * hn, h]"""

        intermediate_shape = (
            num_splits,
            num_attention_heads_per_partition,
            hidden_size_per_attention_head,
        ) + input_shape[1:]

        t = t.view(*intermediate_shape)
        t = t.transpose(0, 1).contiguous()
    else:
        """[np * hn * num_splits, h]
        -->(view) [np, hn, num_splits, h]
        -->(tranpose) [np, num_splits, hn, h]
        -->(view) [np * num_splits * hn, h]"""

        intermediate_shape = (
            num_attention_heads_per_partition,
            hidden_size_per_attention_head,
            num_splits,
        ) + input_shape[1:]

        t = t.view(*intermediate_shape)
        t = t.transpose(1, 2).contiguous()
    t = t.view(*input_shape)

    return t


def fix_query_key_value_ordering(model, checkpoint_version):
    """Fix up query/key/value matrix ordering if checkpoint
    version is smaller than 2.0
    """
    if checkpoint_version < 2.0:
        if isinstance(model, list):
            assert len(model) == 1
            model = model[0]
        for name, param in model.named_parameters():
            if name.endswith((".query_key_value.weight", ".query_key_value.bias")):
                if checkpoint_version == 0:
                    fixed_param = _transpose_first_dim(param.data, 3, True, model)
                elif checkpoint_version == 1.0:
                    fixed_param = _transpose_first_dim(param.data, 3, False, model)
                else:
                    print_rank_0(f"Invalid checkpoint version {checkpoint_version}.")
                    sys.exit()
                param.data.copy_(fixed_param)
            if name.endswith((".key_value.weight", ".key_value.bias")):
                if checkpoint_version == 0:
                    fixed_param = _transpose_first_dim(param.data, 2, True, model)
                elif checkpoint_version == 1.0:
                    fixed_param = _transpose_first_dim(param.data, 2, False, model)
                else:
                    print_rank_0(f"Invalid checkpoint version {checkpoint_version}.")
                    sys.exit()
                param.data.copy_(fixed_param)
        print_rank_0(
            " successfully fixed query-key-values ordering for checkpoint version {}".format(checkpoint_version)
        )


def _get_non_persistent_iteration(non_persistent_global_dir, cfg: ConfigContainer, checkpointing_context=None):
    if cfg.checkpoint_config.non_persistent_ckpt_type is None:
        return -1
    elif cfg.checkpoint_config.non_persistent_ckpt_type == "global":
        train_state_filename = get_checkpoint_train_state_filename(non_persistent_global_dir, prefix=TRACKER_PREFIX)
        if os.path.isfile(train_state_filename):
            train_state = read_train_state(train_state_filename)
            iteration = train_state.step
            # if train_state.release:
            #     raise RuntimeError("Non-persistent checkpoint can't be a release checkpoint")
        else:
            iteration = -1
            print_rank_0("WARNING: could not find the metadata file {}".format(train_state_filename))
            print_rank_0("    will not load any non-persistent checkpoint")
        return iteration
    elif cfg.checkpoint_config.non_persistent_ckpt_type == "local":
        return checkpointing_context["local_checkpoint_manager"].find_latest()
    else:
        assert (
            False
        ), f"Please use local or global non-persistent checkpoints(got: {cfg.checkpoint_config.non_persistent_ckpt_type})"


def _load_non_persistent_base_checkpoint(
    non_persistent_global_dir,
    cfg: ConfigContainer,
    rank0,
    sharded_state_dict,
    non_persistent_iteration,
    checkpointing_context=None,
):
    """Load the base state_dict from a non-persistent distributed checkpoint.
    Depending on the non_persistent_ckpt_type, different logic may be required.
    """
    assert cfg.checkpoint_config.non_persistent_ckpt_type is not None
    if cfg.checkpoint_config.non_persistent_ckpt_type == "global":
        if not rank0:
            print_rank_0(f"Loading from a non-persistent checkpoint (non-persistent iter {non_persistent_iteration})")
        return _load_global_dist_base_checkpoint(
            non_persistent_global_dir,
            cfg,
            rank0,
            sharded_state_dict,
            non_persistent_iteration,
            False,
            checkpointing_context=checkpointing_context,
        )
    elif cfg.checkpoint_config.non_persistent_ckpt_type == "local":
        intermediate_state_dict, checkpoint_name = checkpointing_context["local_checkpoint_manager"].load()
        state_dict = intermediate_state_dict.to_state_dict(
            sharded_state_dict,
            algo=cfg.checkpoint_config.non_persistent_local_ckpt_algo,
            parallelization_group=mpu.get_data_parallel_group(with_context_parallel=True),
        )
        return state_dict, checkpoint_name, False, CheckpointType.LOCAL
    else:
        assert (
            False
        ), f"Please use local or global non-persistent checkpoints(got: {cfg.checkpoint_config.non_persistent_ckpt_type})"


def _load_global_dist_base_checkpoint(
    load_dir, cfg: ConfigContainer, rank0, sharded_state_dict, iteration, release, checkpointing_context=None
):
    """Load the base state_dict from the given directory containing the global distributed checkpoint"""
    if rank0:
        checkpoint_name = find_checkpoint_rank_0(load_dir, iteration, release)
        state_dict = dist_checkpointing.load_common_state_dict(checkpoint_name)
        return state_dict, checkpoint_name, release, CheckpointType.GLOBAL

    if sharded_state_dict is None:
        assert not cfg.checkpoint_config.auto_detect_ckpt_format and not use_dist_ckpt(
            cfg.checkpoint_config.ckpt_format
        ), (
            cfg.checkpoint_config.auto_detect_ckpt_format,
            use_dist_ckpt(cfg.checkpoint_config.ckpt_format),
        )
        raise RuntimeError(
            "Detected load from a distributed checkpoint, but neither --use-dist-ckpt nor --auto-detect-ckpt-format is set."
        )

    checkpoint_name = get_checkpoint_name(load_dir, iteration, release, return_base_dir=True)
    load_strategy = get_default_load_sharded_strategy(checkpoint_name)
    # NOTE: `cfg.checkpoint_config.fully_parallel_load` applies to both persistent and non-persistent checkpoints.
    if cfg.checkpoint_config.fully_parallel_load:
        load_strategy = FullyParallelLoadStrategyWrapper(
            load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
        )
    if checkpointing_context is not None:
        checkpointing_context["load_strategy"] = load_strategy
    state_dict = dist_checkpointing.load(
        sharded_state_dict, checkpoint_name, load_strategy, strict=cfg.checkpoint_config.dist_ckpt_strictness
    )
    return state_dict, checkpoint_name, release, CheckpointType.GLOBAL


def _load_base_checkpoint(
    load_dir,
    cfg: ConfigContainer,
    rank0=False,
    sharded_state_dict=None,
    checkpointing_context=None,
):
    """Load the base state_dict from the given directory

    If rank0 is true, just loads rank 0 checkpoint, ignoring arguments.
    """
    # Try to load non-persistent checkpoint first
    non_persistent_global_dir = (
        cfg.checkpoint_config.non_persistent_global_ckpt_dir
        if cfg.checkpoint_config.non_persistent_global_ckpt_dir or load_dir is None
        else os.path.join(load_dir, _NON_PERSISTENT_CKPT_SUBDIR)
    )
    non_persistent_iteration = _get_non_persistent_iteration(non_persistent_global_dir, cfg, checkpointing_context)
    iteration, release = -1, False
    tracker_filename = "because load directory is not defined"
    if load_dir is not None:
        tracker_filename = get_checkpoint_train_state_filename(load_dir, prefix=TRACKER_PREFIX)
        if os.path.isfile(tracker_filename):
            train_state = read_train_state(tracker_filename)
            iteration = train_state.step
            # release = train_state.release
    if non_persistent_iteration != -1:  # there is a non-persistent checkpoint
        if non_persistent_iteration >= iteration:
            return _load_non_persistent_base_checkpoint(
                non_persistent_global_dir,
                cfg,
                rank0,
                sharded_state_dict,
                non_persistent_iteration,
                checkpointing_context,
            )
        else:
            print_rank_0("WARNING: non-persistent checkpoints are older than persistent checkpoint")

    # Otherwise we are dealing with global checkpoints
    # If no tracker file, return nothing
    if iteration == -1:
        if not rank0:
            print_rank_0("WARNING: could not find the metadata file {}".format(tracker_filename))
            print_rank_0("    will not load any checkpoints and will start from random")
        # Conditionally exit if checkpoint not found.
        if cfg.checkpoint_config.exit_on_missing_checkpoint:
            print_rank_0(">> '--exit-on-missing-checkpoint' set ... exiting. <<")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            sys.exit()

        return None, "", False, None

    # Determine the type of the checkpoint
    checkpoint_name = get_checkpoint_name(load_dir, iteration, release, return_base_dir=True)
    is_dist_ckpt = dist_checkpointing.check_is_distributed_checkpoint(checkpoint_name)
    if not rank0:
        dist_infix = "distributed " if is_dist_ckpt else ""
        if release:
            print_rank_0(f" loading release {dist_infix}checkpoint from {load_dir}")
        else:
            print_rank_0(f" loading {dist_infix}checkpoint from {load_dir} at iteration {iteration}")

    # Handle global distributed checkpoint
    if is_dist_ckpt:
        return _load_global_dist_base_checkpoint(
            load_dir, cfg, rank0, sharded_state_dict, iteration, release, checkpointing_context=checkpointing_context
        )
    else:
        raise RuntimeError("Only distributed checkpoint format is supported")


def fix_fp8_params_lose_precision_when_loading_dist_ckpt(state_dict):
    """
    When "--fp8-param-gather" and "--use-dist-ckpt" are both enabled, the state dict read from
    dist-checkpoint loses precision (the weights read from checkpoint go through the process of
    bf16/fp16 -> fp8 -> bf16/fp16). This function is implemented to solve this problem.
    When "--fp8-param-gather" is disabled, this function doesn't modify anything.
    """
    for key in state_dict.keys():
        if key.startswith("model"):
            for _, sharded_tensor in state_dict[key].items():
                if is_float8tensor(sharded_tensor.data):
                    sharded_tensor.data = sharded_tensor.data.from_float8().cpu()


def load_checkpoint(
    state: GlobalState,
    model,
    optimizer,
    opt_param_scheduler,
    strict=True,
    checkpointing_context=None,
    skip_load_to_model_and_opt=False,
):
    """Load a model checkpoint and return the iteration.
    strict (bool): whether to strictly enforce that the keys in
        :attr:`state_dict` of the checkpoint match the names of
        parameters and buffers in model.
    skip_load_to_model_and_opt (bool): whether to call `load_state_dict`
        for :attr:`model` and :attr:`optimizer`. In case of running FSDP2
        or other torch features that uses DTensor in state dict, the tensors
        are already loaded in-place by `_load_base_checkpoint`.
    """
    cfg = state.cfg
    load_dir = cfg.checkpoint_config.load

    # Finetuning directories
    pretrained_dir = cfg.checkpoint_config.pretrained_checkpoint
    if pretrained_dir is not None and not checkpoint_exists(load_dir):
        print_rank_0(
            f"Checkpoint file not found in load directory {load_dir} attempting to finetune with checkpoint in {pretrained_dir}"
        )
        load_dir = pretrained_dir
        if not checkpoint_exists(load_dir):
            raise FileNotFoundError("No checkpoint found in load directory or pretrained directory")
        cfg.checkpoint_config.finetune = True

    model = unwrap_model(model)

    load_kwargs = {}
    is_dist_ckpt = False
    if (
        cfg.checkpoint_config.auto_detect_ckpt_format
        or use_dist_ckpt(cfg.checkpoint_config.ckpt_format)
        or cfg.checkpoint_config.non_persistent_save_interval is not None
    ):
        state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
            load_dir,
            cfg,
            rank0=True,
            checkpointing_context=checkpointing_context,
        )
        run_config = read_run_config(get_checkpoint_run_config_filename(checkpoint_name))

        is_dist_ckpt = ckpt_type == CheckpointType.LOCAL or dist_checkpointing.check_is_distributed_checkpoint(
            checkpoint_name
        )
        if is_dist_ckpt:
            # TODO: Make read_run_config() return a ConfigContainer object
            ckpt_tp_pp = (
                run_config["model_config"]["tensor_model_parallel_size"],
                run_config["model_config"]["pipeline_model_parallel_size"],
                run_config["model_config"].get("encoder_tensor_model_parallel_size", 0),
                run_config["model_config"].get("encoder_pipeline_model_parallel_size", 0),
            )
            run_tp_pp = (
                cfg.model_config.tensor_model_parallel_size,
                cfg.model_config.pipeline_model_parallel_size,
                getattr(cfg.model_config, "encoder_tensor_model_parallel_size", 0),
                getattr(cfg.model_config, "encoder_pipeline_model_parallel_size", 0),
            )
            mismatch_msg = "(TP, PP, encoder TP, encoder PP) mismatch after resume ({} vs {} from checkpoint)".format(
                run_tp_pp, ckpt_tp_pp
            )

            # Determine if RNG state will be loaded
            if (
                ckpt_tp_pp == run_tp_pp
                and not release
                and not cfg.checkpoint_config.finetune
                and cfg.checkpoint_config.load_rng
                and run_config["checkpoint_config"]["save_rng"]
            ):
                gen_sd_rng_state = get_rng_state(
                    data_parallel_random_init=cfg.rng_config.data_parallel_random_init, use_dist_ckpt=True
                )  # we can load the rng state
            else:
                gen_sd_rng_state = None
                if ckpt_tp_pp != run_tp_pp:
                    print_rank_0("{}: RNG state will be ignored".format(mismatch_msg))

            optim_sd_kwargs = dict(is_loading=True)
            # Determine if optimizer state will be loaded
            if (
                not release
                and not cfg.checkpoint_config.finetune
                and cfg.checkpoint_config.load_optim
                and run_config["checkpoint_config"]["save_optim"]
            ):
                gen_sd_optim = optimizer
                gen_sd_opt_param_scheduler = opt_param_scheduler

                if cfg.optimizer_config.use_distributed_optimizer:
                    optim_sd_kwargs["sharding_type"] = (
                        "fully_sharded_model_space"
                        if run_config["checkpoint_config"]["fully_parallel_save"]
                        else "dp_zero_gather_scatter"
                    )
                    # This is for backwards-compatibility. Can be removed once 'fully_sharded_bucket_space' loading is removed
                    for maybe_dist_opt_optim_state in (state_dict["optimizer"], *state_dict["optimizer"].values()):
                        if "param_state_sharding_type" in maybe_dist_opt_optim_state:
                            if maybe_dist_opt_optim_state["param_state_sharding_type"] == "fully_sharded_bucket_space":
                                print_rank_0(
                                    "Detected deprecated `fully_sharded_bucket_space` DistributedOptimizer checkpoint format"
                                )
                                optim_sd_kwargs["sharding_type"] = maybe_dist_opt_optim_state[
                                    "param_state_sharding_type"
                                ]
                            break

                    if ckpt_tp_pp != run_tp_pp and optim_sd_kwargs["sharding_type"] != "fully_sharded_model_space":
                        raise RuntimeError(
                            f"{mismatch_msg}: not supported for DistributedOptimizer with sharding type {optim_sd_kwargs['sharding_type']}."
                            f" Please use `--ckpt-fully-parallel-save` flag during checkpoint saving."
                        )
            else:
                gen_sd_optim = None
                gen_sd_opt_param_scheduler = None

            # Determine if rerun state will be loaded
            if (
                ckpt_tp_pp == run_tp_pp
                and not release
                and not cfg.checkpoint_config.finetune
                and "rerun_state_machine" in state_dict
            ):
                rerun_state_machine = get_rerun_state_machine()
                gen_sd_rerun_state = rerun_state_machine.state_dict(data_iterator=None, ckpt_format="torch_dist")
            else:
                gen_sd_rerun_state = None
                if ckpt_tp_pp != run_tp_pp:
                    print_rank_0("{}: Rerun state will be ignored".format(mismatch_msg))

            # [ModelOpt]: IMPORTANT! Restoring modelopt_state (sharded or not) must be performed
            # after the model instance has been created and before _load_base_checkpoint is called.
            if has_nvidia_modelopt:
                if ckpt_type == CheckpointType.LOCAL:
                    print_rank_0("WARNING: Local checkpointing does not support nvidia_modelopt.")
                elif ckpt_type == CheckpointType.GLOBAL:
                    restore_modelopt_state(model, state_dict)
                else:
                    restore_sharded_modelopt_state(model, checkpoint_name)

            # [ModelOpt]: Initial loading from non-resume sharded checkpoint to a Distillation Model
            # will result in key mismatch with loss modules potentially containing parameters, since
            # it requires generating a state_dict before loading. Here we hide those modules if present.
            with contextlib.ExitStack() as stack:  # Allows multiple context managers for each model shard
                if cfg.checkpoint_config.finetune and hasattr(model[0], "hide_loss_modules"):
                    for m in model:
                        stack.enter_context(m.hide_loss_modules())
                load_kwargs["sharded_state_dict"] = generate_state_dict(
                    cfg,
                    model,
                    gen_sd_optim,
                    gen_sd_opt_param_scheduler,
                    gen_sd_rng_state,
                    use_dist_ckpt=True,
                    optim_sd_kwargs=optim_sd_kwargs,
                    rerun_state=gen_sd_rerun_state,
                )

            # When "--fp8-param-gather" is disabled, this function doesn't modify anything.
            fix_fp8_params_lose_precision_when_loading_dist_ckpt(load_kwargs["sharded_state_dict"])

    state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
        load_dir, cfg, rank0=False, checkpointing_context=checkpointing_context, **load_kwargs
    )

    # Checkpoint not loaded.
    if state_dict is None:
        # Iteration and num_floating_point_operations_so_far default to 0.
        return 0, 0

    # Set checkpoint version.
    set_checkpoint_version(state_dict.get("checkpoint_version", 0))

    # Check arguments.
    assert state.train_state.consumed_train_samples == 0
    assert state.train_state.skipped_train_samples == 0
    assert state.train_state.consumed_valid_samples == 0

    state.train_state = read_train_state(get_checkpoint_train_state_filename(checkpoint_name))
    # Set iteration.
    if cfg.checkpoint_config.finetune or release:
        state.train_state.step = 0

    if not cfg.checkpoint_config.finetune:
        # check_checkpoint_args(checkpoint_args)
        update_num_microbatches(consumed_samples=state.train_state.consumed_train_samples, verbose=True)

    # Model.
    if not skip_load_to_model_and_opt:
        if len(model) == 1:
            model[0].load_state_dict(state_dict["model"], strict=strict)
        else:
            for i in range(len(model)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                model[i].load_state_dict(state_dict["model%d" % i], strict=strict)

    # Fix up query/key/value matrix ordering if needed.
    checkpoint_version = get_checkpoint_version()
    print_rank_0(f" checkpoint version {checkpoint_version}")
    fix_query_key_value_ordering(model, checkpoint_version)

    # Optimizer.
    if not release and not cfg.checkpoint_config.finetune and cfg.checkpoint_config.load_optim:
        try:
            # Load state dict.
            if (
                not skip_load_to_model_and_opt
                and optimizer is not None
                and not getattr(optimizer, "is_stub_optimizer", False)
            ):
                optimizer.load_state_dict(state_dict["optimizer"])

            # Load distributed optimizer's custom parameter state.
            # For distributed checkpoint it's already loaded in load_state_dict above
            # if cfg.optimizer_config.use_distributed_optimizer and not is_dist_ckpt:
            #     # NOTE: this is a manual read of the tracker file.
            #     # This code should not be reached when reading from a non_persistent checkpoint
            #     assert not is_dist_ckpt
            #     tracker_filename = get_checkpoint_train_state_filename(load_dir, prefix="latest")
            #     iteration, release = read_train_state(tracker_filename)
            #     model_checkpoint_name = get_checkpoint_name(load_dir, iteration, release)
            #     optim_checkpoint_name = get_distributed_optimizer_checkpoint_name(model_checkpoint_name)
            #     optimizer.load_parameter_state(
            #         optim_checkpoint_name, update_legacy_format=cfg.checkpoint_config.ckpt_convert_update_legacy_dist_opt_format
            #     )

            # Load scheduler.
            if opt_param_scheduler is not None:
                if "lr_scheduler" in state_dict:  # backward compatbility
                    opt_param_scheduler.load_state_dict(state_dict["lr_scheduler"])
                else:
                    opt_param_scheduler.load_state_dict(state_dict["opt_param_scheduler"])
        except KeyError as e:
            print_rank_0(
                "Unable to load optimizer from checkpoint {}. "
                "Specify --no-load-optim or --finetune to prevent "
                "attempting to load the optimizer state, "
                "exiting ...".format(checkpoint_name)
            )
            raise e
    else:
        if (cfg.model_config.fp16 or cfg.model_config.bf16) and optimizer is not None:
            optimizer.reload_model_params()

    # rerun state
    try:
        if "rerun_state_machine" in state_dict:
            get_rerun_state_machine().load_state_dict(state_dict["rerun_state_machine"])
    except Exception as e:
        print(f"Unable to restore RerunMachine from checkpoint: {e}")
        sys.exit()

    # rng states.
    if not release and not cfg.checkpoint_config.finetune and cfg.checkpoint_config.load_rng:
        try:
            if "rng_state" in state_dict:
                # access rng_state for data parallel rank
                if cfg.rng_config.data_parallel_random_init:
                    rng_state = state_dict["rng_state"][mpu.get_data_parallel_rank()]
                else:
                    rng_state = state_dict["rng_state"][0]
                random.setstate(rng_state["random_rng_state"])
                np.random.set_state(rng_state["np_rng_state"])
                torch.set_rng_state(rng_state["torch_rng_state"])
                torch.cuda.set_rng_state(rng_state["cuda_rng_state"])
                # Check for empty states array
                if not rng_state["rng_tracker_states"]:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(rng_state["rng_tracker_states"])
            else:  # backward compatability
                random.setstate(state_dict["random_rng_state"])
                np.random.set_state(state_dict["np_rng_state"])
                torch.set_rng_state(state_dict["torch_rng_state"])
                torch.cuda.set_rng_state(state_dict["cuda_rng_state"])
                # Check for empty states array
                if not state_dict["rng_tracker_states"]:
                    raise KeyError
                tensor_parallel.get_cuda_rng_tracker().set_states(state_dict["rng_tracker_states"])
        except KeyError:
            print_rank_0(
                "Unable to load rng state from checkpoint {}. "
                "Specify --no-load-rng or --finetune to prevent "
                "attempting to load the rng state, "
                "exiting ...".format(checkpoint_name)
            )
            sys.exit()

    # Some utilities want to load a checkpoint without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0(
        f"  successfully loaded checkpoint from {load_dir} "
        f"[ t {mpu.get_tensor_model_parallel_rank() + 1}/{mpu.get_tensor_model_parallel_world_size()}, "
        f"p {mpu.get_pipeline_model_parallel_rank() + 1}/{mpu.get_pipeline_model_parallel_world_size()} ] "
        f"at iteration {state.train_state.step}"
    )

    # Additional callback for wandb (last rank)
    if not torch.distributed.is_initialized() or is_last_rank():
        wandb_utils.on_load_checkpoint_success(checkpoint_name, load_dir, state.wandb_logger)

    torch.cuda.empty_cache()

    if state.train_state.step > 0:
        # Notify FT that a checkpoint was loaded.
        is_local_chkpt = ckpt_type == CheckpointType.LOCAL
        fault_tolerance.on_checkpoint_loaded(is_local_chkpt=is_local_chkpt, global_state=state)

    return state.train_state.step, state.train_state.floating_point_operations_so_far
