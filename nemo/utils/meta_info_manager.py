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
from typing import Any, Dict, Optional
import torch

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
    # Minimal configuration - required fields only
    init_config = {
        # Required fields (from OneLoggerConfig) - no defaults
        "application_name": "nemo-application",
        "session_tag_or_fn": os.environ.get("SLURM_JOB_NAME", "nemo-run"),
        # Important fields with defaults - provide if available from config
        "enable_for_current_rank": _should_enable_for_current_rank(),
        # Error handling strategy - use DISABLE_QUIETLY_AND_REPORT_METRIC_ERROR to prevent
        # telemetry errors from crashing the training application
        "error_handling_strategy": "propagate_exceptions",
    }

    return init_config

def _get_base_telemetry_config(
    trainer: Any,
    job_name: str,
    global_batch_size: int,
    seq_length: int,
) -> Dict[str, Any]:
    """Generate base configuration for OneLogger training telemetry.
    
    This function provides the common configuration needed for both NeMo v1 and v2.
    It extracts basic training information from trainer object and uses provided
    batch size and sequence length values.
    
    Args:
        trainer: PyTorch Lightning trainer instance
        job_name: Job name for the experiment
        global_batch_size: Global batch size (calculated by version-specific function)
        seq_length: Sequence length (calculated by version-specific function)
        
    Returns:
        Dictionary containing base training telemetry configuration
    """
    # Extract values from trainer
    job_name = os.environ.get('SLURM_JOB_NAME', job_name)
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    max_steps = getattr(trainer, 'max_steps', 1)
    # Use hardcoded value for log_every_n_steps instead of getting from trainer
    log_every_n_steps = 10  # Most common default value based on NeMo recipes
    micro_batch_size = global_batch_size // world_size
    # Get PERF_VERSION_TAG from environment
    perf_version_tag = os.environ.get('PERF_VERSION_TAG', '0.0.0')

    # Calculate performance tag
    perf_tag = f"{job_name}_{perf_version_tag}_bf{global_batch_size}_se{seq_length}_ws{world_size}"
    
    # Calculate train samples target
    train_samples_target = max_steps * global_batch_size

    # Base training telemetry configuration
    base_config = {
        # Performance tag (REQUIRED in TrainingTelemetryConfig)
        "perf_tag_or_fn": perf_tag,
        # World size and batch information (REQUIRED in TrainingTelemetryConfig)
        "world_size_or_fn": world_size,
        "global_batch_size_or_fn": global_batch_size,
        "micro_batch_size_or_fn": micro_batch_size,
        "seq_length_or_fn": seq_length,
        # Training targets
        "train_iterations_target_or_fn": max_steps,
        "train_samples_target_or_fn": train_samples_target,
        # Logging frequency
        "log_every_n_train_iterations": log_every_n_steps,
    }
    
    return base_config


def get_nemo_v1_telemetry_config(
    trainer: Any,
    job_name: str,
    exp_manager_config: Any,
) -> Dict[str, Any]:
    """Generate NeMo v1 specific configuration for OneLogger training telemetry.
    
    This function provides NeMo v1 specific configuration by extracting values from
    the exp_manager_config object and trainer object.
    
    Args:
        trainer: PyTorch Lightning trainer instance
        job_name: Job name for the experiment
        exp_manager_config: ExpManagerConfig from NeMo v1 (required)
        
    Returns:
        Dictionary containing NeMo v1 training telemetry configuration
    """
    # Calculate batch size and sequence length for NeMo v1
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Get global batch size from trainer's dataloader (most reliable source)
    global_batch_size = 1  # Default fallback
    if hasattr(trainer, 'train_dataloader') and trainer.train_dataloader is not None:
        try:
            dataloader = trainer.train_dataloader
            if hasattr(dataloader, 'batch_size'):
                # Calculate global batch size: batch_size * world_size * accumulate_grad_batches
                micro_batch_size = dataloader.batch_size
                accumulate_grad_batches = getattr(trainer, 'accumulate_grad_batches', 1)
                global_batch_size = micro_batch_size * world_size * accumulate_grad_batches

    # Get sequence length from datamodule (most reliable source)
    seq_length = 1  # Default fallback
    if hasattr(trainer, 'datamodule') and trainer.datamodule is not None:
        try:
            datamodule = trainer.datamodule
            if hasattr(datamodule, 'seq_length'):
                seq_length = datamodule.seq_length
            elif hasattr(datamodule, 'max_len'):
                seq_length = datamodule.max_len

    # Get base configuration with calculated values
    config = _get_base_telemetry_config(
        trainer=trainer, 
        job_name=job_name, 
        global_batch_size=global_batch_size,
        seq_length=seq_length,
    )
    
    # NeMo v1: Extract from exp_manager_config
    is_save_checkpoint_enabled = getattr(exp_manager_config, 'create_checkpoint_callback', True)
    
    # Extract checkpoint strategy from callback params if available
    checkpoint_params = getattr(exp_manager_config, 'checkpoint_callback_params', None)
    save_checkpoint_strategy = "async"  # Default for NeMo v1
    if checkpoint_params and hasattr(checkpoint_params, 'async_save'):
        save_checkpoint_strategy = "async" if checkpoint_params.async_save else "sync"

    # Extract validation configuration from trainer object
    # Use trainer.check_val_every_n_epoch to determine if validation is enabled (NeMo v1)
    val_check_interval = getattr(trainer, 'check_val_every_n_epoch', -1)
    is_validation_iterations_enabled = val_check_interval > 0
    
    # Add NeMo v1 specific configuration
    config.update({
        # Telemetry feature flags
        "is_validation_iterations_enabled_or_fn": is_validation_iterations_enabled,
        "is_save_checkpoint_enabled_or_fn": is_save_checkpoint_enabled,
        "save_checkpoint_strategy": save_checkpoint_strategy,
    })
    
    return config


def get_nemo_v2_telemetry_config(
    trainer: Any,
    job_name: str,
    nemo_logger_config: Any,
    data: Any,
) -> Dict[str, Any]:
    """Generate NeMo v2 specific configuration for OneLogger training telemetry.
    
    This function provides NeMo v2 specific configuration by extracting values from
    the trainer callbacks and nemo_logger_config object.
    
    Args:
        trainer: PyTorch Lightning trainer instance
        job_name: Job name for the experiment
        nemo_logger_config: NeMoLogger config from NeMo v2 (required)
        data: Data module from NeMo v2 (required)
        
    Returns:
        Dictionary containing NeMo v2 training telemetry configuration
    """
    # NeMo v2: Extract batch size and sequence length from data module (most reliable source)
    global_batch_size = 1  # Default fallback
    seq_length = 1  # Default fallback

    if data is not None:
        global_batch_size = data.global_batch_size
        seq_length = data.seq_length
    
    # Get base configuration with calculated values
    config = _get_base_telemetry_config(
        trainer=trainer, 
        job_name=job_name, 
        global_batch_size=global_batch_size,
        seq_length=seq_length,
    )
    
    # Check if checkpoint callback is present
    # First check if nemo_logger_config has a checkpoint callback configured
    is_save_checkpoint_enabled = False
    if hasattr(nemo_logger_config, 'ckpt') and nemo_logger_config.ckpt is not None:
        is_save_checkpoint_enabled = True
    else:
        # Fallback to checking trainer callbacks
        checkpoint_callbacks = [cb for cb in trainer.callbacks if 'Checkpoint' in type(cb).__name__]
        is_save_checkpoint_enabled = len(checkpoint_callbacks) > 0

    # Check if validation is enabled based on trainer configuration
    # Use trainer.val_check_interval to determine if validation is enabled (NeMo v2)
    val_check_interval = getattr(trainer, 'val_check_interval', -1)
    is_validation_iterations_enabled = val_check_interval > 0
 
    # Determine save checkpoint strategy based on trainer strategy and checkpoint configuration
    save_checkpoint_strategy = "async" if hasattr(trainer, 'strategy') and trainer.strategy.async_save else "sync"
    # Add NeMo v2 specific configuration
    config.update({
        # Telemetry feature flags
        "is_validation_iterations_enabled_or_fn": is_validation_iterations_enabled,
        "is_save_checkpoint_enabled_or_fn": is_save_checkpoint_enabled,
        "save_checkpoint_strategy": save_checkpoint_strategy,
    })

    return config


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
