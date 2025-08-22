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

import json
import logging
import os
from functools import partial
from typing import Any, Dict, Optional, Tuple

import nemo_run as run
from rich.table import Table

from nemo.collections import llm
from nemo.collections.llm.tools.auto_configurator import AutoConfigurator, generate_configs
from nemo.collections.llm.tools.auto_configurator.core.performance_helper import set_primary_perf_configs
from nemo.collections.llm.tools.autotuner.args import AutoTuneArgs
from nemo.collections.llm.tools.autotuner.core.display import _display_configs_table
from nemo.collections.llm.tools.autotuner.core.utils import (
    _load_args_from_config_dir,
    console,
    extract_gpu_specs_unified,
    get_args_file_path,
    get_supported_models,
    update_args_with_generation_metadata,
    validate_all_configs,
)
from nemo.lightning.resume import AutoResume

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate(**kwargs):
    """Generate AutoTune configurations for NeMo pretraining."""
    console.print(f"Generating AutoTune configurations for model: [bold]{kwargs['model']}[/bold]")

    args = AutoTuneArgs(**kwargs)

    try:
        result = generate_recipe_configs(args)

        console.print("[yellow]Validating configuration parameters...[/yellow]")
        is_valid, error_msg = validate_all_configs(args)
        if not is_valid:
            console.print("[red]Configuration validation failed:[/red]")
            console.print(f"   {error_msg}")
            raise ValueError(f"Configuration validation failed: {error_msg}")

        console.print("[green]Configuration validation passed![/green]")

        args_file_path = get_args_file_path(args.model, kwargs['config_dir'])
        args.save_to_file(args_file_path)
        console.print(f"[blue]Arguments saved to: {args_file_path}[/blue]")

        console.print("[yellow]Generating configurations with performance optimizations...[/yellow]")

        update_args_with_generation_metadata(args.model, result, kwargs['config_dir'])
        console.print(f"[blue]Metadata and objects saved to: {args_file_path}[/blue]")

        console.print("[green]Configurations generated successfully with performance optimizations![/green]")
        console.print(f"Saved to: {os.path.join(kwargs['config_dir'], args.model)}")
        console.print(f"Generated {result['num_configs_generated']} configurations")

        memory_analysis = result.get('memory_analysis', {})
        if memory_analysis:
            oom_configs = [name for name, analysis in memory_analysis.items() if analysis.get("will_oom", False)]
            safe_configs = [name for name, analysis in memory_analysis.items() if not analysis.get("will_oom", False)]

            console.print("\n[cyan]Memory Analysis Summary:[/cyan]")
            console.print(f"Configurations that will run safely: {len(safe_configs)}")
            if oom_configs:
                console.print(f"⚠ Configurations flagged with potential CUDA OOM: {len(oom_configs)}")
                console.print(f"[yellow]Flagged configs: \n {', '.join(oom_configs)}[/yellow]")
                console.print("[dim]These will be SKIPPED during 'lep autotune run' (use --run-all to force)[/dim]")
            console.print("[blue]Use 'lep autotune list-configs' to see detailed memory analysis[/blue]")

        if result['base_config_matches']:
            console.print(
                f"[blue]Found {len(result['base_config_matches'])} matching configurations: {', '.join(result['base_config_matches'])}[/blue]"
            )

        return result

    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise
    except Exception as e:
        console.print(f"[red]Error generating configurations: {e}[/red]")
        logger.error(f"Configuration generation failed: {e}")
        raise


# ========== SIMPLIFIED MEMORY ESTIMATION ==========


def _calculate_model_weights_memory(
    model_config: Any,
    tp_size: int,
    pp_size: int,
    ep_size: int,
    bytes_per_param: int,
) -> Tuple[float, float, float]:
    """
    Calculate model weights memory for layers and embeddings.

    Returns:
        Tuple of (layer_memory_gb, embedding_memory_gb, total_model_memory_gb)
    """

    # Extract model architecture from config with None handling
    def safe_getattr(obj, attr, default):
        """Get attribute value, return default if attribute doesn't exist or is None"""
        value = getattr(obj, attr, default)
        return default if value is None else value

    num_layers = safe_getattr(model_config, 'num_layers', 32)
    hidden_size = safe_getattr(model_config, 'hidden_size', 4096)
    num_attention_heads = safe_getattr(model_config, 'num_attention_heads', 32)
    vocab_size = safe_getattr(model_config, 'vocab_size', 32000)
    ffn_hidden_size = safe_getattr(model_config, 'ffn_hidden_size', hidden_size * 4)
    num_moe_experts = safe_getattr(model_config, 'num_moe_experts', None)
    moe_router_topk = safe_getattr(model_config, 'moe_router_topk', 1)
    moe_ffn_hidden_size = safe_getattr(model_config, 'moe_ffn_hidden_size', ffn_hidden_size)

    # Check if this is a MoE model
    is_moe_model = num_moe_experts is not None and num_moe_experts > 1

    # Calculate parameters per layer
    if is_moe_model:
        # MoE model: each expert has its own FFN parameters
        # Attention parameters are shared across experts
        attention_params_per_layer = (
            hidden_size * hidden_size * 3  # QKV projection
            + hidden_size * hidden_size  # Output projection
            + hidden_size * 2  # Layer norm parameters
        )

        # MoE FFN parameters per expert
        moe_ffn_params_per_expert = (
            hidden_size * moe_ffn_hidden_size  # Up projection
            + moe_ffn_hidden_size * hidden_size  # Down projection
            + hidden_size * 2  # Layer norm parameters
        )

        # Total FFN parameters (only active experts contribute to memory per GPU)
        active_experts_per_gpu = min(moe_router_topk, num_moe_experts // max(ep_size, 1))
        ffn_params_per_layer = moe_ffn_params_per_expert * active_experts_per_gpu

        params_per_layer = attention_params_per_layer + ffn_params_per_layer
    else:
        # Standard transformer: FFN parameters
        ffn_params_per_layer = (
            hidden_size * ffn_hidden_size  # Up projection
            + ffn_hidden_size * hidden_size  # Down projection
            + hidden_size * 2  # Layer norm parameters
        )

        # Attention parameters
        attention_params_per_layer = (
            hidden_size * hidden_size * 3  # QKV projection
            + hidden_size * hidden_size  # Output projection
            + hidden_size * 2  # Layer norm parameters
        )

        params_per_layer = attention_params_per_layer + ffn_params_per_layer

    # Add embedding parameters (shared across pipeline stages)
    embedding_params = vocab_size * hidden_size

    # Calculate memory for each component with proper parallelism distribution
    # Layer parameters: distributed by both TP and PP
    layer_params = params_per_layer * num_layers
    layer_memory_gb = (layer_params * bytes_per_param) / (1024**3)
    layer_memory_gb /= tp_size  # Sharded by tensor parallelism
    layer_memory_gb /= pp_size  # Distributed by pipeline parallelism

    # Embedding parameters: distributed by both TP and PP
    embedding_memory_gb = (embedding_params * bytes_per_param) / (1024**3)
    embedding_memory_gb /= tp_size  # Sharded by tensor parallelism
    embedding_memory_gb /= pp_size  # Shared across pipeline stages

    # Total model memory
    total_model_memory_gb = layer_memory_gb + embedding_memory_gb

    return layer_memory_gb, embedding_memory_gb, total_model_memory_gb


def _calculate_optimizer_memory(
    model_config: Any,
    tp_size: int,
    pp_size: int,
    ep_size: int,
    bytes_per_param: int,
) -> float:
    """Calculate optimizer memory requirements."""

    # Extract model architecture
    def safe_getattr(obj, attr, default):
        value = getattr(obj, attr, default)
        return default if value is None else value

    num_layers = safe_getattr(model_config, 'num_layers', 32)
    hidden_size = safe_getattr(model_config, 'hidden_size', 4096)
    ffn_hidden_size = safe_getattr(model_config, 'ffn_hidden_size', hidden_size * 4)
    vocab_size = safe_getattr(model_config, 'vocab_size', 32000)
    num_moe_experts = safe_getattr(model_config, 'num_moe_experts', None)
    moe_router_topk = safe_getattr(model_config, 'moe_router_topk', 1)
    moe_ffn_hidden_size = safe_getattr(model_config, 'moe_ffn_hidden_size', ffn_hidden_size)

    is_moe_model = num_moe_experts is not None and num_moe_experts > 1

    # Calculate parameters per layer
    if is_moe_model:
        attention_params_per_layer = hidden_size * hidden_size * 3 + hidden_size * hidden_size + hidden_size * 2
        moe_ffn_params_per_expert = (
            hidden_size * moe_ffn_hidden_size + moe_ffn_hidden_size * hidden_size + hidden_size * 2
        )
        # Properly handle expert parallelism
        active_experts_per_gpu = min(moe_router_topk, num_moe_experts // max(ep_size, 1))
        ffn_params_per_layer = moe_ffn_params_per_expert * active_experts_per_gpu
        params_per_layer = attention_params_per_layer + ffn_params_per_layer
    else:
        ffn_params_per_layer = hidden_size * ffn_hidden_size + ffn_hidden_size * hidden_size + hidden_size * 2
        attention_params_per_layer = hidden_size * hidden_size * 3 + hidden_size * hidden_size + hidden_size * 2
        params_per_layer = attention_params_per_layer + ffn_params_per_layer

    # Add embedding parameters
    embedding_params = vocab_size * hidden_size

    # AdamW optimizer: momentum + variance for each parameter
    optimizer_states_per_param = 2  # momentum + variance

    # Optimizer memory for layer parameters (distributed by TP and PP)
    layer_params = params_per_layer * num_layers
    layer_optimizer_memory_gb = (layer_params * optimizer_states_per_param * bytes_per_param) / (1024**3)
    layer_optimizer_memory_gb /= tp_size
    layer_optimizer_memory_gb /= pp_size

    # Optimizer memory for embedding parameters (distributed by TP and PP)
    embedding_optimizer_memory_gb = (embedding_params * optimizer_states_per_param * bytes_per_param) / (1024**3)
    embedding_optimizer_memory_gb /= tp_size
    embedding_optimizer_memory_gb /= pp_size

    return layer_optimizer_memory_gb + embedding_optimizer_memory_gb


def _calculate_gradient_memory(
    model_config: Any,
    tp_size: int,
    pp_size: int,
    ep_size: int,
    bytes_per_param: int,
) -> float:
    """Calculate gradient memory requirements."""

    # Extract model architecture
    def safe_getattr(obj, attr, default):
        value = getattr(obj, attr, default)
        return default if value is None else value

    num_layers = safe_getattr(model_config, 'num_layers', 32)
    hidden_size = safe_getattr(model_config, 'hidden_size', 4096)
    ffn_hidden_size = safe_getattr(model_config, 'ffn_hidden_size', hidden_size * 4)
    vocab_size = safe_getattr(model_config, 'vocab_size', 32000)
    num_moe_experts = safe_getattr(model_config, 'num_moe_experts', None)
    moe_router_topk = safe_getattr(model_config, 'moe_router_topk', 1)
    moe_ffn_hidden_size = safe_getattr(model_config, 'moe_ffn_hidden_size', ffn_hidden_size)

    is_moe_model = num_moe_experts is not None and num_moe_experts > 1

    # Calculate parameters per layer
    if is_moe_model:
        attention_params_per_layer = hidden_size * hidden_size * 3 + hidden_size * hidden_size + hidden_size * 2
        moe_ffn_params_per_expert = (
            hidden_size * moe_ffn_hidden_size + moe_ffn_hidden_size * hidden_size + hidden_size * 2
        )
        # Properly handle expert parallelism
        active_experts_per_gpu = min(moe_router_topk, num_moe_experts // max(ep_size, 1))
        ffn_params_per_layer = moe_ffn_params_per_expert * active_experts_per_gpu
        params_per_layer = attention_params_per_layer + ffn_params_per_layer
    else:
        ffn_params_per_layer = hidden_size * ffn_hidden_size + ffn_hidden_size * hidden_size + hidden_size * 2
        attention_params_per_layer = hidden_size * hidden_size * 3 + hidden_size * hidden_size + hidden_size * 2
        params_per_layer = attention_params_per_layer + ffn_params_per_layer

    # Add embedding parameters
    embedding_params = vocab_size * hidden_size

    # Gradients: same size as parameters
    # Gradient memory for layer parameters (distributed by TP and PP)
    layer_params = params_per_layer * num_layers
    layer_gradient_memory_gb = (layer_params * bytes_per_param) / (1024**3)
    layer_gradient_memory_gb /= tp_size
    layer_gradient_memory_gb /= pp_size

    # Gradient memory for embedding parameters (distributed by TP and PP)
    embedding_gradient_memory_gb = (embedding_params * bytes_per_param) / (1024**3)
    embedding_gradient_memory_gb /= tp_size
    embedding_gradient_memory_gb /= pp_size

    return layer_gradient_memory_gb + embedding_gradient_memory_gb


def _calculate_activation_memory(
    model_config: Any,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    vp_size: int,
    micro_batch_size: int,
    seq_length: int,
    bytes_per_param: int,
) -> float:
    """Calculate activation memory requirements."""

    # Extract model architecture
    def safe_getattr(obj, attr, default):
        value = getattr(obj, attr, default)
        return default if value is None else value

    num_layers = safe_getattr(model_config, 'num_layers', 32)
    hidden_size = safe_getattr(model_config, 'hidden_size', 4096)
    num_moe_experts = safe_getattr(model_config, 'num_moe_experts', None)

    # Check if this is a MoE model
    is_moe_model = num_moe_experts is not None and num_moe_experts > 1

    # Virtual pipeline parallelism calculations
    # VP divides layers across virtual stages within each physical pipeline stage
    layers_per_pp_stage = num_layers // pp_size

    # effective micro batch size (accounting for virtual pipeline)
    # in VP, we process smaller chunks of the micro batch through each virtual stage
    # max(1, micro_batch_size // vp_size) to ensure we don't get 0
    effective_mbs = max(1, micro_batch_size // vp_size)

    # Base activation memory per layer (conservative estimates)
    if is_moe_model:
        # MoE models: slightly higher due to expert routing overhead
        base_activation_per_layer = hidden_size * 0.1  # 10% of hidden size per layer
    else:
        # Standard transformers: conservative estimate
        # Use a simpler, more predictable scaling approach
        base_activation_per_layer = hidden_size * 0.08  # 8% of hidden size per layer

    # Scale with effective batch size and sequence length
    # Conservative scaling factors
    batch_scale = min(effective_mbs, 8)  # Cap at 8 to avoid overestimation
    seq_scale = min(seq_length / 8192, 2.0)  # Scale relative to 8k seq, cap at 2x

    # Total activation memory (conservative)
    # Note: Activations are typically stored in bf16/fp16 (2 bytes) regardless of model parameter precision
    activation_memory_gb = (base_activation_per_layer * num_layers * batch_scale * seq_scale * 2) / (
        1024**3
    )  # Use 2 bytes (bf16/fp16) instead of bytes_per_param for activations

    # Apply parallelism distribution
    # Activations are distributed across tensor, pipeline, and context parallelism
    activation_memory_gb /= max(tp_size, 1)  # Sharded by tensor parallelism
    activation_memory_gb /= max(pp_size, 1)  # Distributed by pipeline parallelism
    activation_memory_gb /= max(cp_size, 1)  # Distributed by context parallelism

    # Virtual pipeline parallelism effects on activation memory:
    if vp_size > 1:
        # VP reduces peak activation memory due to smaller effective batch size
        vp_activation_factor = 0.9  # 10% reduction due to smaller chunks
        activation_memory_gb = activation_memory_gb * vp_activation_factor

        # Additional memory for virtual pipeline buffers
        # Each virtual stage needs buffer space for inter-stage communication
        # Use 2 bytes per parameter (bf16/fp16) for activations, not bytes_per_param
        vp_buffer_memory_gb = (hidden_size * effective_mbs * seq_length * 2 * vp_size * 0.02) / (
            1024**3
        )  # 2% of hidden size as buffer per virtual stage
        activation_memory_gb += vp_buffer_memory_gb

    # Additional MoE scaling if needed
    if is_moe_model:
        activation_memory_gb *= 1.2  # 20% overhead for MoE routing

    return activation_memory_gb


def _calculate_pipeline_overhead(
    pp_size: int,
    vp_size: int,
) -> float:
    """Calculate pipeline parallelism overhead."""
    pipeline_overhead_gb = 0.0
    if pp_size > 1:
        # Base pipeline overhead (reduced for high-parallelism configurations)
        pipeline_overhead_gb = 0.05  # Reduced from 0.1 to 0.05 (50MB base overhead per pipeline stage)

        # Virtual pipeline adds additional overhead
        if vp_size > 1:
            # Each virtual stage adds some overhead for stage management (reduced)
            pipeline_overhead_gb += 0.02 * vp_size  # Reduced from 0.05 to 0.02 (20MB per virtual stage)

    return pipeline_overhead_gb


def estimate_model_memory_usage(
    model_config: Any,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    ep_size: int,
    vp_size: int,
    micro_batch_size: int,
    seq_length: int,
) -> float:
    """
    Estimate memory usage for a model configuration using actual model architecture.
    This function orchestrates the calculation by calling specialized helper functions.
    """

    # Get precision from config
    params_dtype = getattr(model_config, 'params_dtype', None)
    bytes_per_param = 4 if 'float32' in str(params_dtype) else 2  # fp32=4, fp16/bf16=2

    # Extract model architecture for logging
    def safe_getattr(obj, attr, default):
        value = getattr(obj, attr, default)
        return default if value is None else value

    num_layers = safe_getattr(model_config, 'num_layers', 32)
    hidden_size = safe_getattr(model_config, 'hidden_size', 4096)
    num_attention_heads = safe_getattr(model_config, 'num_attention_heads', 32)
    num_moe_experts = safe_getattr(model_config, 'num_moe_experts', None)
    is_moe_model = num_moe_experts is not None and num_moe_experts > 1

    # Calculate memory components using specialized functions
    layer_memory_gb, embedding_memory_gb, model_memory_gb = _calculate_model_weights_memory(
        model_config, tp_size, pp_size, ep_size, bytes_per_param
    )

    optimizer_memory_gb = _calculate_optimizer_memory(model_config, tp_size, pp_size, ep_size, bytes_per_param)

    gradient_memory_gb = _calculate_gradient_memory(model_config, tp_size, pp_size, ep_size, bytes_per_param)

    activation_memory_gb = _calculate_activation_memory(
        model_config, tp_size, pp_size, cp_size, vp_size, micro_batch_size, seq_length, bytes_per_param
    )

    pipeline_overhead_gb = _calculate_pipeline_overhead(pp_size, vp_size)

    # Calculate total memory
    total_memory_gb = (
        model_memory_gb + optimizer_memory_gb + gradient_memory_gb + activation_memory_gb + pipeline_overhead_gb
    )

    # Enhanced logging for transparency
    parallelism_factor = min(1.0, (tp_size * pp_size * cp_size) / 16.0) if not is_moe_model else 1.0
    logger.debug(
        f"Memory estimate (VP-enhanced): Model={num_layers}L/{hidden_size}H/{num_attention_heads}A, MoE={is_moe_model}"
    )
    logger.debug(f"  VP config: {vp_size} virtual stages")
    logger.debug(f"  Parallelism factor: {parallelism_factor:.2f} (TP={tp_size}, PP={pp_size}, CP={cp_size})")
    logger.debug(f"  Layer weights: {layer_memory_gb:.2f}GB, Embedding weights: {embedding_memory_gb:.2f}GB")
    logger.debug(
        f"  Total weights: {model_memory_gb:.2f}GB, Total optimizer: {optimizer_memory_gb:.2f}GB, Total gradients: {gradient_memory_gb:.2f}GB"
    )
    logger.debug(f"  Activations: {activation_memory_gb:.2f}GB")
    logger.debug(f"  Pipeline overhead: {pipeline_overhead_gb:.2f}GB")
    logger.debug(
        f"  Total: {total_memory_gb:.2f}GB, Parallelism: TP={tp_size}, PP={pp_size}, CP={cp_size}, EP={ep_size}, VP={vp_size}"
    )

    return total_memory_gb


def check_cuda_oom_risk(
    config_values: Dict[str, Any],
    resource_shape: str,
    model_name: str,
    model_config: Any,
    safety_margin_gb: float = 0.5,
    memory_per_gpu: Optional[float] = None,
) -> Tuple[bool, str, float, float]:
    """
    Check if a configuration will likely result in CUDA OOM.

    Args:
        config_values: Already-extracted configuration values dict (with 'tp', 'pp', 'cp', etc.)
        resource_shape: Resource shape string like "gpu.8xh200"
        model_name: Name of the model
        safety_margin_gb: Safety margin in GB to leave unused
        memory_per_gpu: Optional custom memory per GPU in GB

    Returns:
        Tuple of (will_oom: bool, reason: str, estimated_usage_gb: float, available_gb: float)
    """

    # Use unified GPU specs extraction
    gpu_type, gpu_count, gpu_memory_gb = extract_gpu_specs_unified(resource_shape, memory_per_gpu)
    available_memory_gb = gpu_memory_gb - safety_margin_gb

    # Extract config values
    tp_size = config_values.get('tp')
    pp_size = config_values.get('pp')
    cp_size = config_values.get('cp')
    ep_size = config_values.get('ep')
    micro_batch_size = config_values.get('mbs')
    seq_length = config_values.get('seq_length')
    vp_size = config_values.get('vp')
    if vp_size is None or vp_size == 'None':
        vp_size = 1
    else:
        vp_size = int(vp_size)

    # use comprehensive memory estimation with actual model config
    estimated_usage_gb = estimate_model_memory_usage(
        model_config=model_config,
        tp_size=tp_size,
        pp_size=pp_size,
        cp_size=cp_size,
        ep_size=ep_size,
        vp_size=vp_size,
        micro_batch_size=micro_batch_size,
        seq_length=seq_length,
    )

    will_oom = estimated_usage_gb > available_memory_gb

    logger.debug(
        f"  Memory: estimated={estimated_usage_gb:.2f}GB, available={available_memory_gb:.2f}GB, will_oom={will_oom}"
    )

    if will_oom:
        reason = (
            f"Conservative estimate ({estimated_usage_gb:.2f} GB) exceeds "
            f"available memory ({available_memory_gb:.2f} GB) on {resource_shape}"
        )
    else:
        reason = f"Configuration likely fits (conservative: {estimated_usage_gb:.2f} GB / {available_memory_gb:.2f} GB available)"

    return will_oom, reason, estimated_usage_gb, gpu_memory_gb


def validate_configurations_memory(
    configs: Dict[str, Any],
    base_config_values: Dict[str, Any],
    resource_shape: str,
    model_name: str,
    model_config: Any,
    memory_per_gpu: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Validate all configurations for potential CUDA OOM issues.

    Args:
        configs: Dictionary of generated configurations
        base_config: Base configuration object
        resource_shape: Resource shape string
        model_name: Model name
        memory_per_gpu: Optional custom memory per GPU in GB

    Returns:
        Dictionary with memory analysis for each configuration
    """
    memory_analysis = {}

    # Base config is always considered valid - skip OOM risk check
    # Still estimate memory for informational purposes
    _, _, usage_gb, total_gb = check_cuda_oom_risk(
        base_config_values, resource_shape, model_name, model_config=model_config, memory_per_gpu=memory_per_gpu
    )

    memory_analysis["base_config"] = {
        "will_oom": False,  # Base config always runs regardless of memory estimation
        "reason": f"Base config - estimated usage: {usage_gb:.2f} GB / {total_gb:.2f} GB available",
        "estimated_usage_gb": usage_gb,
        "total_gpu_memory_gb": total_gb,
        "config_values": base_config_values,
    }

    # Check each generated config using unified extraction call
    for config_name, config_obj in configs.items():
        config_values = extract_all_values(config_name)

        will_oom, reason, usage_gb, total_gb = check_cuda_oom_risk(
            config_values, resource_shape, model_name, model_config=model_config, memory_per_gpu=memory_per_gpu
        )

        memory_analysis[config_name] = {
            "will_oom": will_oom,
            "reason": reason,
            "estimated_usage_gb": usage_gb,
            "total_gpu_memory_gb": total_gb,
            "config_values": config_values,
        }

    return memory_analysis


# ======================== Generate and Save Configs ========================


def generate_recipe_configs(args):
    """
    Generate AutConfigurator recipe configurations.
    Uses unified extraction system.
    Args:
        args: Arguments object with all configuration parameters

    Returns:
        dict: Dictionary containing:
            - base_config: Base configuration object
            - configs: Dictionary of generated configurations
            - runner: AutoConfigurator runner object
            - num_configs_generated: Number of configurations generated
            - base_config_matches: List of configs that match base config
            - memory_analysis: To know which configs will result in CUDA OOM
    """
    is_valid, error_msg = validate_all_configs(args)
    if not is_valid:
        raise ValueError(f"Configuration validation failed: {error_msg}")

    model_class = getattr(llm, args.model, None)
    if model_class is None:
        supported_models = get_supported_models()
        raise ValueError(
            f"Model {args.model} not found in llm module. \n"
            f"Supported models: {', '.join(supported_models)}\n"
            f"For the latest list, check: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes/__init__.py"
        )

    recipe = partial(model_class.pretrain_recipe, num_nodes=args.nodes, num_gpus_per_node=args.gpus_per_node)()
    # Set performance optimizations directly using set_primary_perf_configs
    recipe = set_primary_perf_configs( # this can go in autconf for sure
        recipe=recipe,
        task="pre_train",  # AutoTune is for pretraining
        num_nodes=recipe.trainer.num_nodes,
        num_gpus_per_node=recipe.trainer.devices,
        mbs=recipe.data.micro_batch_size,
        gbs=recipe.data.global_batch_size,
        max_steps=recipe.trainer.max_steps,
        tp_size=getattr(recipe.trainer.strategy, 'tensor_model_parallel_size', 1),
        pp_size=getattr(recipe.trainer.strategy, 'pipeline_model_parallel_size', 1),
        cp_size=getattr(recipe.trainer.strategy, 'context_parallel_size', 1),
        vp_size=getattr(recipe.trainer.strategy, 'virtual_pipeline_model_parallel_size', 1) or 1,
        ep_size=getattr(recipe.trainer.strategy, 'expert_model_parallel_size', 1),
        etp_size=getattr(recipe.trainer.strategy, 'expert_tensor_parallel_size', None),
        enable_cuda_graphs=False,  # Disabled for FSDP compatibility
        use_mcore_fsdp=False,  # Disabled for compatibility with NeMo
        use_user_buffer_registration=None,  # Let the function auto-detect based on model size
        use_sharp=False,  # Enable SHARP for collective communication
        recompute_layers=0,  # No recompute for base config
        activation_offload_layers=0,  # No offload for base config
        keep_fsdp_fp8_transpose_cache=False,
        compute_dtype=getattr(recipe, 'compute_dtype', 'bf16') or 'bf16',
        fp8_recipe=None,  # Use default FP8 recipe
        recompute_modules=None,  # No selective recompute
        nccl_communicator_config_path=None,  # Use default NCCL config
        model_name=args.model,  # Pass model name for auto-detection
    )

    seq_length = getattr(args, 'seq_length', 8192)
    max_steps = getattr(args, 'max_steps', 10)


    # these things go right before pretraining in autotuner 
    recipe.model.config.seq_length = recipe.data.seq_length = seq_length
    recipe.trainer.max_steps = max_steps
    recipe.trainer.val_check_interval = max_steps
    recipe.trainer.enable_checkpointing = False # stays here
    recipe.trainer.log_every_n_steps = 1 
    recipe.trainer.limit_val_batches = 0 # stays here
    recipe.trainer.strategy.ckpt_async_save = False # stays here
    recipe.resume = run.Config(AutoResume) 
    recipe.log.ckpt.save_last = False # stays here

    base_log_path = args.get_full_logs_path() # stays here
    recipe.log.log_dir = os.path.join(base_log_path, "base_config") # stays here

    num_moe_experts = getattr(recipe.model.config, "num_moe_experts", 0)
    if num_moe_experts and num_moe_experts > 1:
        recipe.trainer.strategy.sequence_parallel = True

    gpu_type, gpu_count, gpu_memory_gb = extract_gpu_specs_unified(
        args.resource_shape, getattr(args, 'memory_per_gpu', None)
    )

    # Initialize Auto Configurator runner
    runner = AutoConfigurator(
        recipe=recipe,
        path_to_logs=base_log_path,
        gpu_memory_gb=gpu_memory_gb,
        tensor_parallel_sizes=args.tensor_parallel_sizes,
        pipeline_parallel_sizes=args.pipeline_parallel_sizes,
        context_parallel_sizes=args.context_parallel_sizes,
        expert_parallel_sizes=args.expert_parallel_sizes,
        virtual_pipeline_model_parallel_sizes=args.virtual_pipeline_model_parallel_sizes,
        micro_batch_sizes=args.micro_batch_sizes,
        global_batch_sizes=args.global_batch_sizes,
        max_model_parallel_size=args.max_model_parallel_size,
        min_model_parallel_size=args.min_model_parallel_size,
        max_steps_per_run=args.max_steps_per_run,
        max_minutes_per_run=args.max_minutes_per_run,
        num_tokens_in_b=args.num_tokens_in_b,
        vocab_size=args.vocab_size,
        calculate_model_size=False,
    )

    base_config, configs, base_configuration_matches = generate_configs(runner, args.resource_shape)
    num_configs_generated = len(configs)

    base_config_values = {
        'tp': base_config.trainer.strategy.tensor_model_parallel_size,
        'pp': base_config.trainer.strategy.pipeline_model_parallel_size,
        'cp': base_config.trainer.strategy.context_parallel_size,
        'ep': base_config.trainer.strategy.expert_model_parallel_size,
        'mbs': base_config.data.micro_batch_size,
        'vp': base_config.trainer.strategy.virtual_pipeline_model_parallel_size,
        'seq_length': base_config.data.seq_length,
        'gbs': base_config.data.global_batch_size,
        'nodes': args.nodes,
        'model_size_b': base_config.model.config.model_size,
        'precision': base_config.trainer.strategy.precision,
    }

    model_family = list(configs.keys())[0].split('_')[0]

    base_config_generated_name = (
        f"{model_family}_{base_config_values['model_size_b']}b_{base_config_values['nodes']}nodes_"
        f"tp_{base_config_values['tp']}_pp_{base_config_values['pp']}_cp_{base_config_values['cp']}_ep_{base_config_values['ep']}_mbs_{base_config_values['mbs']}_vp_{base_config_values['vp']}_seq_{base_config_values['seq_length']}_gbs_{base_config_values['gbs']}"
    )

    args.metadata['base_config_generated_name'] = base_config_generated_name

    logger.info("Performing CUDA OOM analysis for all configurations...")
    # Get model config from the recipe for accurate memory estimation
    if not hasattr(base_config, 'model') or not hasattr(base_config.model, 'config'):
        raise ValueError("Model config not available in base_config. Cannot perform accurate memory estimation.")

    model_config = base_config.model.config
    memory_analysis = validate_configurations_memory(
        configs,
        base_config_values,
        args.resource_shape,
        args.model,
        model_config=model_config,
        memory_per_gpu=getattr(args, 'memory_per_gpu', None),
    )

    oom_configs = [name for name, analysis in memory_analysis.items() if analysis["will_oom"]]
    safe_configs = [name for name, analysis in memory_analysis.items() if not analysis["will_oom"]]

    logger.info("Memory Analysis Summary:")
    logger.info(f"  Total configurations: {len(memory_analysis)}")
    logger.info(f"  Safe configurations: {len(safe_configs)}")
    logger.info(f"  Potential OOM configurations: {len(oom_configs)}")

    if oom_configs:
        logger.warning(f"  Number of configurations with potential OOM: {len(oom_configs)}")

    # Save generated configs and check for matches
    save_generated_configs(args, base_config, configs)

    # Update args with generation metadata (serialization will handle complex objects)
    generation_result = {
        'base_config': base_config,
        'configs': configs,
        'runner': runner,
        'num_configs_generated': num_configs_generated,
        'base_config_matches': [],
        'memory_analysis': {},
        'config_names': list(configs.keys()),
    }
    args.update_metadata(generation_result)

    base_config_path = os.path.join(args.config_dir, args.model, "base_config.json")
    generated_configs_dir = os.path.join(args.config_dir, args.model)

    has_matches = len(base_configuration_matches) > 0
    if has_matches:
        for bcm in base_configuration_matches:
            logger.info(f"Config '{bcm}' matches base config - will be flagged as base config equivalent")
        logger.info(
            f"Found {len(base_configuration_matches)} matching configs. Using original log_dir: {recipe.log.log_dir}"
        )
    else:
        recipe.log.log_dir = os.path.join(base_log_path, args.metadata['base_config_generated_name'])
        logger.info(f"No matching configs found. Updated log_dir to: {recipe.log.log_dir}")

    # Update the generation result with final metadata
    generation_result['base_config_matches'] = base_configuration_matches
    generation_result['memory_analysis'] = memory_analysis
    args.update_metadata(generation_result)

    # Save the updated args.json with final metadata
    args.save_to_file(os.path.join(args.config_dir, args.model, "args.json"))

    return generation_result


def save_generated_configs(args, base_config, configs: Dict):
    """
    Save generated configurations to disk.

    Args:
        args: AutoTuneArgs object containing config_dir and model
        base_config: Base configuration object
        configs: Dictionary of configuration objects
    """
    os.makedirs(args.config_dir, exist_ok=True)
    model_dir = os.path.join(args.config_dir, args.model)
    os.makedirs(model_dir, exist_ok=True)

    # Save base_config.json
    with open(os.path.join(model_dir, "base_config.json"), "w") as f:
        json.dump(base_config.__dict__, f, indent=4, default=str)

    # Save individual config files
    for config_name, recipe in configs.items():
        with open(os.path.join(model_dir, f"{config_name}.json"), "w") as f:
            json.dump(recipe.__dict__, f, indent=4, default=str)


# ======================== List Configs / Models ============================================


def list_configs(config_dir, model):
    """List generated AutoTune configurations with detailed status."""
    try:
        args = _load_args_from_config_dir(config_dir, model)
        model_config_dir = os.path.join(config_dir, args.model)

        console.print(f"Configurations for model: [bold]{args.model}[/bold]")
        console.print(f"Location: {model_config_dir}")
        _display_configs_table(model_config_dir, args.model)

    except Exception:
        console.print(
            "[link]Please check: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes/__init__.py[/link]\n"
        )
        raise


def list_models():
    """List all supported models for AutoTune."""
    try:
        supported_models = get_supported_models()

        console.print("[green]Supported AutoTune Models:[/green]")
        console.print(
            "[link]Reference: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes/__init__.py[/link]"
        )
        console.print()

        table = Table(show_header=True, show_lines=False, title="Available Models")
        table.add_column("Model Name", style="green")
        table.add_column("Description", style="cyan")

        for model in supported_models:
            description = "Language model"
            if "nemotron" in model.lower():
                description = "NVIDIA Nemotron model"
            elif "llama" in model.lower():
                description = "LLaMA-based model"
            elif "mistral" in model.lower():
                description = "Mistral model"
            elif "mixtral" in model.lower():
                description = "Mixtral MoE model"

            table.add_row(model, description)

        console.print(table)
        console.print(f"\n[green]Total: {len(supported_models)} supported models[/green]")

        return supported_models

    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]")
        console.print(
            "[link]Please check: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes/__init__.py[/link]"
        )
