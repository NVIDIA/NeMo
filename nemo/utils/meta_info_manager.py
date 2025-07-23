#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""MetaInfoManager module for handling experiment metadata configuration."""

import os
import uuid
from typing import Any, Dict, Optional

import nemo_run.config as run
import torch
from omegaconf import OmegaConf

from nemo.utils.import_utils import safe_import_from

enable_onelogger = True


def get_onelogger_init_config() -> Dict[str, Any]:
    """Generate minimal configuration for OneLogger initialization.

    This function provides the absolute minimal configuration needed for OneLogger initialization.
    It only includes the required fields and uses defaults for everything else to avoid
    dependencies on exp_manager during early import.

    Returns:
        Dictionary containing minimal initialization configuration
    """
    # Minimal configuration - required fields + important fields with defaults
    init_config = {
        # Required fields (from OneLoggerConfig) - no defaults
        "application_name": "nemo-application",
        "session_tag_or_fn": os.environ.get("SLURM_JOB_NAME", "nemo-run"),
        # Important fields with defaults - provide if available from config
        "enable_for_current_rank": _should_enable_for_current_rank(),
        "is_train_iterations_enabled_or_fn": True,
        "is_validation_iterations_enabled_or_fn": True,
        "is_test_iterations_enabled_or_fn": False,
        "is_save_checkpoint_enabled_or_fn": True,
        # Skip fields with safe defaults or handled in TrainingLoopConfig:
        # - is_log_throughput_enabled_or_fn: defaults to False (requires additional config)
        "save_checkpoint_strategy": "async",  # TODO: need to remove this in nv-one-logger side
        # - summary_data_schema_version_or_fn: defaults to "1.0.0"
    }

    return init_config


def get_onelogger_training_loop_config(
    trainer: Any,
    job_name: str,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """Generate configuration for OneLogger training loop updates.

    This function provides the configuration needed for updating OneLogger with
    training loop specific information. It extracts data from trainer, model, and data objects
    with fallbacks to config paths when objects are not available.

    Args:
        trainer: PyTorch Lightning trainer instance
        job_name: Job name for the experiment
        model: PyTorch Lightning model instance (optional)

    Returns:
        Dictionary containing training loop configuration
    """

    # Extract values from trainer
    world_size = getattr(trainer, 'world_size', 1) * getattr(trainer, 'num_nodes', 1)
    max_steps = getattr(trainer, 'max_steps', 1)
    log_every_n_steps = getattr(trainer, 'log_every_n_steps', 10)

    # Extract values from model or provided global_batch_size
    if model is not None:
        global_batch_size = getattr(model, 'global_batch_size', 1)
        micro_batch_size = global_batch_size // world_size or getattr(model, 'micro_batch_size', 1)
        seq_length = getattr(model, 'seq_length', 1)
    else:
        get_current_global_batch_size, HAVE_MCORE_MBATCH_CALCULATOR = safe_import_from(
            "megatron.core.num_microbatches_calculator", "get_current_global_batch_size"
        )
        global_batch_size = get_current_global_batch_size()
        micro_batch_size = global_batch_size // world_size
        seq_length = getattr(trainer, 'datamodule', {}).get('seq_length', 1)  # TODO: need to test this

    # Training loop specific configuration (TrainingLoopConfig fields)
    training_loop_config = {
        # Performance tag (REQUIRED in TrainingLoopConfig)
        "perf_tag_or_fn": (
            f"{job_name}_{os.environ.get('PERF_VERSION_TAG', '0.0.0')}_" f"{global_batch_size}_" f"{world_size}"
        ),
        # World size and batch information (REQUIRED in TrainingLoopConfig)
        "world_size_or_fn": world_size,
        "global_batch_size_or_fn": global_batch_size,
        # Optional training loop parameters
        "micro_batch_size_or_fn": micro_batch_size,
        "seq_length_or_fn": seq_length,
        # Training targets
        "train_iterations_target_or_fn": max_steps,
        "train_samples_target_or_fn": max_steps * global_batch_size,
        # Logging frequency
        "log_every_n_train_iterations": log_every_n_steps,
    }

    return training_loop_config


def _should_enable_for_current_rank() -> bool:
    """Determine if OneLogger should be enabled for the current rank.

    In distributed training, typically only rank 0 (or the last rank) should
    enable OneLogger to avoid duplicate telemetry data.

    Returns:
        True if OneLogger should be enabled for the current rank, False otherwise
    """
    try:
        # Check if distributed training is initialized
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            # Enable for rank 0 or the last rank (common pattern)
            return rank == 0 or rank == world_size - 1
        else:
            # Single process training - always enable
            return True
    except Exception:
        # If there's any error checking distributed state, default to True
        return False
