import os
import sys
import json
import logging
from typing import Dict, Any, Optional, Tuple
from functools import partial
import nemo_run as run

from nemo.collections import llm
from rich.table import Table

from nemo.collections.llm.tools.autotuner.args import AutoTuneArgs
from nemo.collections.llm.tools.autotuner.core.display import _display_memory_analysis, _display_configs_table, display_performance_analysis

from nemo.collections.llm.tools.autotuner.core.utils import (
    extract_gpu_specs_unified, extract_all_values, get_supported_models,
    get_args_file_path, update_args_with_generation_metadata,
    create_log_dir_name, check_config_matches,
    validate_all_configs, _load_args_from_config_dir,
    logger, console
)

from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.collections.llm.tools.auto_configurator import AutoConfigurator, generate_configs, get_results
from nemo.lightning.resume import AutoResume
from scripts.performance.helpers import set_primary_perf_configs, set_perf_optimization_configs
from scripts.performance.utils import get_comm_overlap_callback_idx

def set_performance_optimizations_aligned_with_nemo(recipe, args):
    """
    Set performance optimizations using NeMo's standard set_primary_perf_configs function.
    This ensures we use the exact same optimization logic as the performant scripts.
    
    Note: User buffer configurations are handled per-config in autotuner/core/training_config.py
    to ensure each configuration gets appropriate settings based on its TP/PP/CP values.
    """
    
    # Extract GPU specs for optimization parameters
    gpu_type, _, _ = extract_gpu_specs_unified(args.resource_shape, getattr(args, 'memory_per_gpu', None))
    compute_dtype = getattr(recipe, 'compute_dtype', 'bf16') or 'bf16'
    
    # Get current parallelism settings from recipe
    tp_size = getattr(recipe.trainer.strategy, 'tensor_model_parallel_size', 1)
    pp_size = getattr(recipe.trainer.strategy, 'pipeline_model_parallel_size', 1)
    cp_size = getattr(recipe.trainer.strategy, 'context_parallel_size', 1)
    vp_size = getattr(recipe.trainer.strategy, 'virtual_pipeline_model_parallel_size', 1)
    if vp_size is None:
        vp_size = 1
    
    # Use NeMo's standard performance configuration function
    recipe = set_primary_perf_configs(
        recipe=recipe,
        task="pre_train",  # AutoTune is for pretraining
        num_nodes=recipe.trainer.num_nodes,
        num_gpus_per_node=recipe.trainer.devices,
        mbs=recipe.data.micro_batch_size,
        gbs=recipe.data.global_batch_size,
        max_steps=recipe.trainer.max_steps,
        tp_size=tp_size,
        pp_size=pp_size,
        cp_size=cp_size,
        vp_size=vp_size,
        ep_size=getattr(recipe.trainer.strategy, 'expert_model_parallel_size', 1),
        etp_size=getattr(recipe.trainer.strategy, 'expert_tensor_parallel_size', None),
        enable_cuda_graphs=False,  # Disabled for FSDP compatibility
        use_mcore_fsdp=False,  # Disabled for compatibility with NeMo
        use_user_buffer_registration=True,  # Enable for performance
        use_sharp=False,  # Enable SHARP for collective communication
        recompute_layers=0,  # No recompute for base config
        activation_offload_layers=0,  # No offload for base config
        compute_dtype=compute_dtype,
        fp8_recipe=None,  # Use default FP8 recipe
        recompute_modules=None,  # No selective recompute
        nccl_communicator_config_path=None,  # Use default NCCL config
    )
    return recipe


def generate(**kwargs):
    """Generate AutoTune configurations for NeMo pretraining."""
    console.print(f"Generating AutoTune configurations for model: [bold]{kwargs['model']}[/bold]")
    
    # print the received values for debugging
    console.print(f"[blue]Received parameters:[/blue]")
    console.print(f"  Global batch sizes: {kwargs['global_batch_sizes']}")
    console.print(f"  Tensor parallel sizes: {kwargs['tensor_parallel_sizes']}")
    console.print(f"  Pipeline parallel sizes: {kwargs['pipeline_parallel_sizes']}")
    console.print(f"  Context parallel sizes: {kwargs['context_parallel_sizes']}")
    console.print(f"  Expert parallel sizes: {kwargs['expert_parallel_sizes']}")
    console.print(f"  Micro batch sizes: {kwargs['micro_batch_sizes']}")
    gpu_type, gpu_count, gpu_memory_gb = extract_gpu_specs_unified(kwargs['resource_shape'], kwargs.get('memory_per_gpu'))
    model_info = extract_all_values(kwargs['model'])
    model_size_b = model_info.get('model_size_b')
    
    console.print(f" Resource: [blue]{kwargs['resource_shape']}[/blue] ({gpu_type.upper()}, {gpu_memory_gb}GB per GPU)")
    if model_size_b:
        console.print(f" Model: [cyan]{kwargs['model']}[/cyan] ({model_size_b}B parameters)")
    else:
        console.print(f" Model: [cyan]{kwargs['model']}[/cyan]")
    
    args = AutoTuneArgs(**kwargs)
    
    try:
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

        result = generate_recipe_configs(args)

        update_args_with_generation_metadata(args.model, result, kwargs['config_dir'])
        console.print(f"[blue]Metadata and objects saved to: {args_file_path}[/blue]")
        
        console.print("[green]Configurations generated successfully with performance optimizations![/green]")
        console.print(f"Saved to: {os.path.join(kwargs['config_dir'], args.model)}")
        console.print(f"Generated {result['num_configs_generated']} configurations")
        
        memory_analysis = result.get('memory_analysis', {})
        if memory_analysis:
            oom_configs = [name for name, analysis in memory_analysis.items() if analysis.get("will_oom", False)]
            safe_configs = [name for name, analysis in memory_analysis.items() if not analysis.get("will_oom", False)]
            
            console.print(f"\n[cyan]Memory Analysis Summary:[/cyan]")
            console.print(f"Configurations that will run safely: {len(safe_configs)}")
            if oom_configs:
                console.print(f"⚠ Configurations flagged with potential CUDA OOM: {len(oom_configs)}")
                console.print(f"[yellow]Flagged configs: \n {', '.join(oom_configs)}[/yellow]")
                console.print(f"[dim]These will be SKIPPED during 'lep autotune run' (use --run-all to force)[/dim]")
            
            console.print(f"\n[blue]All configurations have been generated and saved[/blue]")
            console.print(f"[blue]Use 'lep autotune list-configs' to see detailed memory analysis[/blue]")
        
        if result['base_config_matches']:
            console.print(f"[blue]Found {len(result['base_config_matches'])} matching configurations: {', '.join(result['base_config_matches'])}[/blue]")
        
        return result
        
    except ValueError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise
    except Exception as e:
        console.print(f"[red]Error generating configurations: {e}[/red]")
        logger.error(f"Configuration generation failed: {e}")
        raise

# ========== SIMPLIFIED MEMORY ESTIMATION ==========

def estimate_model_memory_usage(
    model_config: Any,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    ep_size: int,
    vp_size: int,
    micro_batch_size: int,
    seq_length: int
) -> float:
    """
    Estimate memory usage for a model configuration using actual model architecture.
    """
    # Extract model architecture from config with None handling
    def safe_getattr(obj, attr, default):
        """Get attribute value, return default if attribute doesn't exist or is None"""
        value = getattr(obj, attr, default)
        return default if value is None else value
    
    num_layers = safe_getattr(model_config, 'num_layers', 32)
    hidden_size = safe_getattr(model_config, 'hidden_size', 4096)
    num_attention_heads = safe_getattr(model_config, 'num_attention_heads', 32)
    num_query_groups = safe_getattr(model_config, 'num_query_groups', num_attention_heads)  # For grouped query attention
    vocab_size = safe_getattr(model_config, 'vocab_size', 32000)
    ffn_hidden_size = safe_getattr(model_config, 'ffn_hidden_size', hidden_size * 4)
    num_moe_experts = safe_getattr(model_config, 'num_moe_experts', None)
    moe_router_topk = safe_getattr(model_config, 'moe_router_topk', 1)
    moe_ffn_hidden_size = safe_getattr(model_config, 'moe_ffn_hidden_size', ffn_hidden_size)
    # Check if this is a MoE model
    is_moe_model = num_moe_experts is not None and num_moe_experts > 1
    
    # Get precision from config
    params_dtype = getattr(model_config, 'params_dtype', None)
    bytes_per_param = 4 if 'float32' in str(params_dtype) else 2  # fp32=4, fp16/bf16=2

    # Effective micro batch size (accounting for virtual pipeline)
    effective_mbs = micro_batch_size // vp_size
    
    # 1. MODEL WEIGHTS MEMORY CALCULATION
    # ===================================
    
    # Calculate parameters per layer
    if is_moe_model:
        # MoE model: each expert has its own FFN parameters
        # Attention parameters are shared across experts
        attention_params_per_layer = (
            hidden_size * hidden_size * 3 +  # QKV projection
            hidden_size * hidden_size +      # Output projection
            hidden_size * 2                  # Layer norm parameters
        )
        
        # MoE FFN parameters per expert
        moe_ffn_params_per_expert = (
            hidden_size * moe_ffn_hidden_size +  # Up projection
            moe_ffn_hidden_size * hidden_size +  # Down projection
            hidden_size * 2                      # Layer norm parameters
        )
        
        # Total FFN parameters (only active experts contribute to memory per GPU)
        active_experts_per_gpu = min(moe_router_topk, num_moe_experts // ep_size)
        ffn_params_per_layer = moe_ffn_params_per_expert * active_experts_per_gpu
        
        params_per_layer = attention_params_per_layer + ffn_params_per_layer
    else:
        # Standard transformer: FFN parameters
        ffn_params_per_layer = (
            hidden_size * ffn_hidden_size +  # Up projection
            ffn_hidden_size * hidden_size +  # Down projection
            hidden_size * 2                  # Layer norm parameters
        )
        
        # Attention parameters
        attention_params_per_layer = (
            hidden_size * hidden_size * 3 +  # QKV projection
            hidden_size * hidden_size +      # Output projection
            hidden_size * 2                  # Layer norm parameters
        )
        
        params_per_layer = attention_params_per_layer + ffn_params_per_layer
    
    # Add embedding parameters (shared across pipeline stages)
    embedding_params = vocab_size * hidden_size
    
    # Total model parameters
    total_params = params_per_layer * num_layers + embedding_params
    
    # Model weights memory (in GB)
    model_memory_gb = (total_params * bytes_per_param) / (1024**3)
    
    # Divide by tensor parallelism (weights are sharded)
    model_memory_gb /= tp_size
    
    # 2. OPTIMIZER MEMORY CALCULATION
    # ===============================
    
    # AdamW optimizer: momentum + variance for each parameter
    # Plus potential additional states for advanced optimizers
    optimizer_states_per_param = 2  # momentum + variance
    optimizer_memory_gb = (total_params * optimizer_states_per_param * bytes_per_param) / (1024**3)
    optimizer_memory_gb /= tp_size
    
    # 3. GRADIENT MEMORY CALCULATION
    # ==============================
    
    # Gradients: same size as parameters
    gradient_memory_gb = (total_params * bytes_per_param) / (1024**3)
    gradient_memory_gb /= tp_size
    
    # 4. ACTIVATION MEMORY CALCULATION (Simplified & Conservative)
    # ============================================================
    
    # Use model size-based heuristics for activation memory
    # This is more generalizable and conservative (underestimates rather than overestimates)
    
    # Base activation memory per layer (conservative estimates)
    if is_moe_model:
        # MoE models: slightly higher due to expert routing overhead
        base_activation_per_layer = hidden_size * 0.1  # 10% of hidden size per layer
    else:
        # Standard transformers: conservative estimate
        base_activation_per_layer = hidden_size * 0.08  # 8% of hidden size per layer
    
    # Scale with effective batch size and sequence length
    # Conservative scaling factors
    batch_scale = min(effective_mbs, 8)  # Cap at 8 to avoid overestimation
    seq_scale = min(seq_length / 8192, 2.0)  # Scale relative to 8k seq, cap at 2x
    
    # Total activation memory (conservative)
    activation_memory_gb = (
        base_activation_per_layer * 
        num_layers * 
        batch_scale * 
        seq_scale * 
        bytes_per_param
    ) / (1024**3)
    
    # Apply parallelism distribution (conservative)
    activation_memory_gb /= max(pp_size, 1)
    activation_memory_gb /= max(cp_size, 1)
    
    # Additional MoE scaling if needed
    if is_moe_model:
        activation_memory_gb *= 1.2  # 20% overhead for MoE routing
    
    # 5. TOTAL MEMORY
    # ===============
    
    total_memory_gb = model_memory_gb + optimizer_memory_gb + gradient_memory_gb + activation_memory_gb
    
    # Enhanced logging for transparency
    logger.debug(f"Memory estimate (simplified): Model={num_layers}L/{hidden_size}H/{num_attention_heads}A, MoE={is_moe_model}")
    logger.debug(f"  Weights: {model_memory_gb:.2f}GB, Optimizer: {optimizer_memory_gb:.2f}GB, Gradients: {gradient_memory_gb:.2f}GB")
    logger.debug(f"  Activations (conservative): {activation_memory_gb:.2f}GB (batch_scale={batch_scale:.1f}, seq_scale={seq_scale:.1f})")
    logger.debug(f"  Total: {total_memory_gb:.2f}GB, Parallelism: TP={tp_size}, PP={pp_size}, CP={cp_size}, EP={ep_size}, VP={vp_size}")
    
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
    precision = config_values.get('precision')

    # Use comprehensive memory estimation with actual model config
    estimated_usage_gb = estimate_model_memory_usage(
        model_config=model_config,
        tp_size=tp_size,
        pp_size=pp_size,
        cp_size=cp_size,
        ep_size=ep_size,
        vp_size=vp_size,
        micro_batch_size=micro_batch_size,
        seq_length=seq_length
    )

    will_oom = estimated_usage_gb > available_memory_gb

    if will_oom:
        reason = (f"Conservative estimate ({estimated_usage_gb:.2f} GB) exceeds "
                 f"available memory ({available_memory_gb:.2f} GB) on {resource_shape}")
    else:
        reason = f"Configuration likely fits (conservative: {estimated_usage_gb:.2f} GB / {available_memory_gb:.2f} GB available)"

    return will_oom, reason, estimated_usage_gb, gpu_memory_gb

def validate_configurations_memory(
    configs: Dict[str, Any],
    base_config: Any,
    resource_shape: str,
    model_name: str,
    model_config: Any,  # Mandatory model config object
    memory_per_gpu: Optional[float] = None
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

    # Check base config using ONE unified extraction call
    base_config_values = extract_all_values(base_config)
    
    # Base config is always considered valid - skip OOM risk check
    # Still estimate memory for informational purposes
    _, _, usage_gb, total_gb = check_cuda_oom_risk(
        base_config_values, resource_shape, model_name, 
        model_config=model_config,
        memory_per_gpu=memory_per_gpu
    )
    
    memory_analysis["base_config"] = {
        "will_oom": False,  # Base config always runs regardless of memory estimation
        "reason": f"Base config - estimated usage: {usage_gb:.2f} GB / {total_gb:.2f} GB available",
        "estimated_usage_gb": usage_gb,
        "total_gpu_memory_gb": total_gb,
        "config_values": base_config_values
    }
    
    # Check each generated config using unified extraction call
    for config_name, config_obj in configs.items():
        config_values = extract_all_values(config_obj)
        
        will_oom, reason, usage_gb, total_gb = check_cuda_oom_risk(
            config_values, resource_shape, model_name,
            model_config=model_config,
            memory_per_gpu=memory_per_gpu
        )
        
        memory_analysis[config_name] = {
            "will_oom": will_oom,
            "reason": reason,
            "estimated_usage_gb": usage_gb,
            "total_gpu_memory_gb": total_gb,
            "config_values": config_values
        }
    
    return memory_analysis

# ======================== Generate and Save Configs ========================

def generate_recipe_configs(args):
    """
    Generate AutoTune recipe configurations.
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

    base_log_path = args.get_full_logs_path()

    recipe = partial(model_class.pretrain_recipe, num_nodes=args.nodes, num_gpus_per_node=args.gpus_per_node)()
    recipe = set_performance_optimizations_aligned_with_nemo(recipe, args)
    
    seq_length = getattr(args, 'seq_length', 8192)
    val_check_interval = getattr(args, 'val_check_interval', 50)
    max_steps = getattr(args, 'max_steps', 10)
    
    recipe.model.config.seq_length = recipe.data.seq_length = seq_length
    recipe.trainer.max_steps = max_steps
    recipe.trainer.val_check_interval = max_steps
    recipe.trainer.enable_checkpointing = False
    recipe.trainer.log_every_n_steps = 1
    recipe.trainer.limit_val_batches = 0
    recipe.trainer.strategy.ckpt_async_save = False
    recipe.resume = run.Config(AutoResume)
    recipe.log.ckpt.save_last = False
    recipe.log.log_dir = os.path.join(base_log_path, "base_config")
    num_moe_experts = getattr(recipe.model.config, "num_moe_experts", 0)
    if num_moe_experts and num_moe_experts > 1:
        recipe.trainer.strategy.sequence_parallel = True

    gpu_type, gpu_count, gpu_memory_gb = extract_gpu_specs_unified(
        args.resource_shape, getattr(args, 'memory_per_gpu', None)
    )

    logger.info(f"Using dynamic log path: {base_log_path}")
    logger.info(f"  Mount path: {args.mount_path}")
    logger.info(f"  Logs subdir: {args.logs_subdir}")
    
    # Initialize Auto Configurator runner
    runner = AutoConfigurator(
        recipe=recipe,
        path_to_logs=base_log_path,
        gpu_memory_gb=gpu_memory_gb,
        tensor_parallel_sizes=args.tensor_parallel_sizes,
        pipeline_parallel_sizes=args.pipeline_parallel_sizes,
        context_parallel_sizes=args.context_parallel_sizes,
        expert_parallel_sizes=args.expert_parallel_sizes,
        virtual_pipeline_parallel_sizes=args.virtual_pipeline_parallel_sizes,
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

    base_config, configs = generate_configs(runner, args.resource_shape)
    num_configs_generated = len(configs)

    logger.info("Performing CUDA OOM analysis for all configurations...")
    # Get model config from the recipe for accurate memory estimation
    if not hasattr(base_config, 'model') or not hasattr(base_config.model, 'config'):
        raise ValueError("Model config not available in base_config. Cannot perform accurate memory estimation.")
    
    model_config = base_config.model.config
    memory_analysis = validate_configurations_memory(
        configs, base_config, args.resource_shape, args.model, 
        model_config=model_config,
        memory_per_gpu=getattr(args, 'memory_per_gpu', None)
    )
    
    oom_configs = [name for name, analysis in memory_analysis.items() if analysis["will_oom"]]
    safe_configs = [name for name, analysis in memory_analysis.items() if not analysis["will_oom"]]
    
    logger.info(f"Memory Analysis Summary:")
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
        'num_configs_generated': len(configs),
        'base_config_matches': [],
        'memory_analysis': {},
        'config_names': list(configs.keys())
    }
    args.update_metadata(generation_result)
    
    base_config_path = os.path.join(args.config_dir, args.model, "base_config.json")
    generated_configs_dir = os.path.join(args.config_dir, args.model)

    config_values = extract_all_values(base_config)
    args.metadata['base_config_generated_name'] = create_log_dir_name(args.model, config_values)

    has_matches, matching_files = check_config_matches(base_config_path, generated_configs_dir)
    
    base_config_matches = []
    if has_matches:
        for matching_file in matching_files:
            config_name = matching_file.replace('.json', '')
            if config_name in configs:
                base_config_matches.append(config_name)
                logger.info(f"Config '{config_name}' matches base config - will be flagged as base config equivalent")
        logger.info(f"Found {len(matching_files)} matching configs. Using original log_dir: {recipe.log.log_dir}")
    else:
        recipe.log.log_dir = os.path.join(base_log_path, args.metadata['base_config_generated_name'])
        logger.info(f"No matching configs found. Updated log_dir to: {recipe.log.log_dir}")

    # Update the generation result with final metadata
    generation_result['base_config_matches'] = base_config_matches
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
        
    except Exception as e:
        console.print("[link]Please check: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes/__init__.py[/link]")
        raise


def list_models():
    """List all supported models for AutoTune."""
    try:
        supported_models = get_supported_models()
        
        console.print("[green]Supported AutoTune Models:[/green]")
        console.print("[link]Reference: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes/__init__.py[/link]")
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
        console.print("[link]Please check: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes/__init__.py[/link]")
        raise