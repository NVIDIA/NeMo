"""
This module provides easy access to NeMo's performance optimization functions around TP communication overlap.
"""

import sys
import os
import logging
from typing import Any, Optional, Dict, Tuple
from nemo.collections.llm.recipes.tp_overlap_configs import userbuffers
import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
import nemo_run as run
from nemo.utils.exp_manager import TimingCallback
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def configure_tp_comm_overlap_intelligently(
    recipe,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    compute_dtype: str,
    fp8_recipe: Optional[str] = None,
    gpu_type: Optional[str] = None,
    hidden_size: Optional[int] = None,
    seq_length: Optional[int] = None,
    micro_batch_size: Optional[int] = None,
) -> Tuple[bool, Optional[Any]]:
    """
    Intelligently configure TP communication overlap based on NeMo performance patterns.
    
    Args:
        recipe: The recipe object
        tp_size: Tensor parallelism size
        pp_size: Pipeline parallelism size  
        cp_size: Context parallelism size
        compute_dtype: Compute dtype (bf16, fp8, etc.)
        fp8_recipe: FP8 recipe type (mxfp8, etc.)
        gpu_type: GPU type (h100, b200, etc.)
        hidden_size: Model hidden size
        seq_length: Sequence length
        micro_batch_size: Micro batch size
    Returns:
        Tuple of (enable_tp_overlap, tp_comm_overlap_cfg)
    """
    
    logger.debug(f"Configuring TP overlap: tp={tp_size}, pp={pp_size}, cp={cp_size}, "
                f"dtype={compute_dtype}, fp8_recipe={fp8_recipe}, gpu={gpu_type}, "
                f"hidden={hidden_size}, seq={seq_length}, mbs={micro_batch_size}")

    if compute_dtype.lower() == "fp8" and fp8_recipe and fp8_recipe.lower() == "mxfp8":
        logger.debug("Disabling TP overlap: FP8 with MXFP8 recipe")
        return False, None

    if all([gpu_type, hidden_size, seq_length, micro_batch_size]):
        logger.debug("Attempting to find exact user buffer match")
        user_buffer_cfg = _find_exact_user_buffer_match(
            gpu_type, compute_dtype, hidden_size, tp_size, cp_size, 
            micro_batch_size, seq_length
        )
        if user_buffer_cfg:
            logger.debug("Found exact user buffer match")
            return True, user_buffer_cfg
        else:
            logger.debug("No exact user buffer match found")
    else:
        logger.debug("Missing parameters for user buffer match: "
                    f"gpu_type={gpu_type}, hidden_size={hidden_size}, "
                    f"seq_length={seq_length}, micro_batch_size={micro_batch_size}")
    logger.debug("Using safe default: TP overlap enabled, no user buffers")
    return True, None


def _find_exact_user_buffer_match(
    gpu_type: str,
    compute_dtype: str,
    hidden_size: int,
    tp_size: int,
    cp_size: int,
    micro_batch_size: int,
    seq_length: int,
) -> Optional[Any]:
    """Find exact user buffer configuration match."""
    if "h200" in gpu_type.lower():
        gpu_type = "h100"
    
    try:
        config_names_to_try = []
        
        if cp_size > 1:
            # Pattern 1: Include CP in name (for CP > 1)
            config_names_to_try.append(
                f"userbuffers_{compute_dtype}_{gpu_type}_h{hidden_size}_tp{tp_size}_cp{cp_size}_mbs{micro_batch_size}_seqlen{seq_length}"
            )
        # Pattern 2: No CP in name (implied CP=1, but also checking for CP > 1)
        config_names_to_try.append(
            f"userbuffers_{compute_dtype}_{gpu_type}_h{hidden_size}_tp{tp_size}_mbs{micro_batch_size}_seqlen{seq_length}"
        )
        logger.debug(f"Trying user buffer patterns: {config_names_to_try}")
        for config_name in config_names_to_try:
            if hasattr(userbuffers, config_name):
                config = getattr(userbuffers, config_name)
                logger.debug(f"Found user buffer config: {config_name}")
                return fdl.cast(run.Config, fdl_dc.convert_dataclasses_to_configs(config))
        logger.debug(f"TP Overlap: Tried configs {config_names_to_try} but none found")
        return None
    except (AttributeError, KeyError) as e:
        logger.warning(f"Error finding user buffer config: {e}")
        return None


def extract_compute_dtype_from_precision(precision):
    """
    Extract compute dtype from precision string.
    
    Args:
        precision: Precision string (e.g., 'bf16-mixed', 'fp16-mixed', 'fp8-mixed')
        
    Returns:
        Compute dtype string ('bf16', 'fp16', 'fp8', or empty string)
    """
    if 'bf16' in precision:
        return 'bf16'
    elif 'fp16' in precision:
        return 'fp16'
    elif 'fp8' in precision:
        return 'fp8'
    return ''


def apply_per_config_tp_comm_overlap_optimization(new_cfg, base_cfg, resource_shape):
    """
    Apply per-config TP communication overlap optimization to a generated config.
    
    Args:
        new_cfg: The generated configuration dictionary to optimize
        base_cfg: Base configuration containing model parameters
        resource_shape: Resource shape from AutoTuneArgs (e.g., 'gpu.8xh100')
        
    Returns:
        The optimized configuration dictionary
    """
    logger.debug(f"Applying TP comm overlap optimization for config: {new_cfg.get('name', 'unknown')}")
    logger.debug(f"Resource shape: {resource_shape}")
    logger.debug(f"TP: {new_cfg.get('tensor_model_parallel_size')}, PP: {new_cfg.get('pipeline_model_parallel_size')}, CP: {new_cfg.get('context_parallel_size')}")
    
    tp_size = new_cfg.get('tensor_model_parallel_size')
    pp_size = new_cfg.get('pipeline_model_parallel_size')
    cp_size = new_cfg.get('context_parallel_size')
    vp_size = new_cfg.get('virtual_pipeline_model_parallel_size')
    micro_batch_size = new_cfg.get('micro_batch_size')
    
    # Calculate DP size for overlap parameter gather
    # Get trainer parameters from the config
    num_nodes = base_cfg.trainer.num_nodes
    num_gpus_per_node = base_cfg.trainer.devices
    
    # Calculate DP size
    dp_size = (num_nodes * num_gpus_per_node) / (tp_size * pp_size * cp_size)
    overlap_param_gather_with_optimizer_step = bool(dp_size > 1 and pp_size > 1 and vp_size > 1)
    wgrad_deferral_limit = 50  # Fixed value as per user request
    
    try:
        hidden_size = base_cfg.model.config.hidden_size
        seq_length = base_cfg.model.config.seq_length
        compute_dtype = ''
        trainer_config = base_cfg.trainer
        
        plugins = trainer_config.plugins
        if hasattr(plugins, 'precision'):
            precision = plugins.precision
            compute_dtype = extract_compute_dtype_from_precision(precision)
        elif isinstance(plugins, list):
            for plugin in plugins:
                if hasattr(plugin, 'precision'):
                    precision = plugin.precision
                    compute_dtype = extract_compute_dtype_from_precision(precision)
                    break

        enable_tp_overlap, tp_comm_overlap_cfg = configure_tp_comm_overlap_intelligently(
            recipe=None,
            tp_size=tp_size,
            pp_size=pp_size,
            cp_size=cp_size,
            compute_dtype=compute_dtype,
            fp8_recipe=None,
            gpu_type=resource_shape,
            hidden_size=hidden_size,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
        )
        
        callbacks = [
            run.Config(TimingCallback()),
            run.Config(GarbageCollectionCallback(gc_interval_train=100, gc_interval_val=100))
        ]
        
        megatron_callback = run.Config(MegatronCommOverlapCallback(
            tp_comm_overlap=enable_tp_overlap,
            tp_comm_overlap_cfg=tp_comm_overlap_cfg,
            tp_comm_bootstrap_backend="nccl",
            defer_embedding_wgrad_compute=True,
            wgrad_deferral_limit=wgrad_deferral_limit,
            overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step
        ))

        callbacks.append(megatron_callback)
        new_cfg['callbacks'] = callbacks
    except Exception as e:
        logger.error(f"Error applying TP comm overlap optimization: {e}")
    return new_cfg


__all__ = [
    'configure_tp_comm_overlap_intelligently',
    'apply_per_config_tp_comm_overlap_optimization',
]