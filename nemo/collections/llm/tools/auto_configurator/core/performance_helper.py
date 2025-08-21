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

"""
Performance helper functions for AutoTuner.
Extracted from NeMo's scripts/performance/helpers.py to avoid external dependencies.
"""

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)


def get_comm_overlap_callback_idx(callbacks):
    """Get the index of the communication overlap callback if it exists."""
    if not callbacks:
        return None

    for idx, callback in enumerate(callbacks):
        if hasattr(callback, '__fn_or_cls__') and 'CommOverlap' in str(callback.__fn_or_cls__):
            return idx
    return None


def set_mcore_fsdp_configs(recipe, comm_overlap_callback_idx: int | None, tp_size: int | None):
    """Set Mcore FSDP related configs."""
    recipe.model.config.init_model_with_meta_device = True
    recipe.trainer.strategy.fsdp = "megatron"
    recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    # At fp32 gradient, `recipe.trainer.strategy.ddp.gradient_reduce_div_fusion` is used for fusion
    if recipe.trainer.plugins.grad_reduce_in_fp32:
        recipe.trainer.strategy.ddp.average_in_collective = False

    recipe.trainer.strategy.ddp.keep_fp8_transpose_cache_when_using_custom_fsdp = False
    recipe.model.config.gradient_accumulation_fusion = False

    if (
        comm_overlap_callback_idx is not None
        and recipe.trainer.callbacks[comm_overlap_callback_idx].defer_embedding_wgrad_compute
    ):
        logger.warning("Disabling deferring embedding wgrad compute because it cannot work with FSDP together.")
        recipe.trainer.callbacks[comm_overlap_callback_idx].defer_embedding_wgrad_compute = False

    return recipe


def set_precision_configs(recipe, compute_dtype: str, fp8_recipe: str | None = None):
    """Set precision related configs."""
    if compute_dtype is None:
        return recipe

    if compute_dtype.lower() == "bf16":
        recipe.optim.config.use_precision_aware_optimizer = True

    if compute_dtype is not None and compute_dtype.lower() == "fp8":
        if fp8_recipe is None:
            fp8_recipe = "ds"

        # Import precision plugins only when needed
        from nemo.collections.llm.recipes.precision.mixed_precision import (
            bf16_with_fp8_current_scaling_mixed,
            bf16_with_fp8_mixed,
            bf16_with_fp8_subchannel_scaling_mixed,
            bf16_with_mxfp8_mixed,
        )

        if fp8_recipe.lower() == "ds":
            recipe.trainer.plugins = bf16_with_fp8_mixed()
        elif fp8_recipe.lower() == "cs":
            recipe.trainer.plugins = bf16_with_fp8_current_scaling_mixed()
            # disable first/last layer bf16 for benchmarking
            recipe.trainer.plugins.first_last_layers_bf16 = False
        elif fp8_recipe.lower() == "mxfp8":
            recipe.trainer.plugins = bf16_with_mxfp8_mixed()
        elif fp8_recipe.lower() == "ss":
            recipe.trainer.plugins = bf16_with_fp8_subchannel_scaling_mixed()

    recipe.trainer.plugins.grad_reduce_in_fp32 = False

    # Enable reuse_grad_buf_for_mxfp8_param_ag for MXFP8 and disable AG overlap
    # because it is not supported with reuse_grad_buf_for_mxfp8_param_ag
    if compute_dtype.lower() == "fp8" and fp8_recipe.lower() == "mxfp8":
        comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
        if comm_overlap_callback_idx is not None:
            recipe.trainer.callbacks[comm_overlap_callback_idx].overlap_param_gather = False
        logger.warning(
            "When using MXFP8, to reduce memory usage, we use reuse_grad_buf_for_mxfp8_param_ag. "
            "Disabling AG overlap because it is not supported with reuse_grad_buf_for_mxfp8_param_ag."
        )

    return recipe


def set_recompute_configs(
    recipe,
    recompute_layers: int,
    activation_offload_layers: int,
    recompute_modules: Optional[List[str]],
):
    """Set activation recomputing and offloading related configs."""
    if recompute_layers > 0:
        recipe.model.config.recompute_granularity = "full"
        recipe.model.config.recompute_method = "block"
        recipe.model.config.recompute_num_layers = recompute_layers

    # Activation cpu offloading
    if activation_offload_layers > 0:
        recipe.model.config.cpu_offloading = True
        recipe.model.config.cpu_offloading_weights = False
        recipe.model.config.cpu_offloading_num_layers = activation_offload_layers

    # Activation recompute configs
    if recompute_modules is not None:
        recipe.model.config.recompute_modules = recompute_modules
        assert (
            recipe.model.config.recompute_granularity == "selective"
        ), "recompute_granularity must be selective when recompute_modules is provided"
        assert (
            recipe.model.config.recompute_num_layers is None
        ), "recompute_num_layers must be None when recompute_modules is provided"

    return recipe


def set_cuda_graph_configs(recipe, enable_cuda_graphs: bool, task: str):
    """Set CUDA graph related configs."""
    recipe.model.config.enable_cuda_graph = enable_cuda_graphs
    recipe.trainer.strategy.use_te_rng_tracker = enable_cuda_graphs

    if (
        task in ["none", "lora"]
        and hasattr(recipe.data, "packed_sequence_specs")
        and recipe.data.packed_sequence_specs is not None
    ):
        recipe.data.packed_sequence_specs.pad_cu_seqlens = enable_cuda_graphs

    return recipe


def set_perf_optimization_configs(
    recipe,
    use_mcore_fsdp: bool,
    enable_cuda_graphs: bool,
    task: str,
    tp_size: int | None,
    compute_dtype: str,
    fp8_recipe: str | None,
    recompute_layers: int,
    activation_offload_layers: int,
    recompute_modules: Optional[List[str]],
    use_fsdp_double_buffer: Optional[bool] = None,
    use_user_buffer_registration: Optional[bool] = None,
    use_sharp: Optional[bool] = None,
    keep_fsdp_fp8_transpose_cache: Optional[bool] = None,
):
    """Set performance optimization related configs."""
    # enable cross entropy fusion with TE kernel
    recipe.model.config.cross_entropy_fusion_impl = "te"

    if use_fsdp_double_buffer:
        assert use_mcore_fsdp == True, "use_fsdp_double_buffer requires use_mcore_fsdp to be True"

    if use_mcore_fsdp and enable_cuda_graphs:
        logger.warning("Currently, cuda graphs are not supported with FSDP. Disabling cuda graphs.")
        enable_cuda_graphs = False

    recipe = set_cuda_graph_configs(recipe, enable_cuda_graphs, task)

    if use_mcore_fsdp:
        comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
        recipe = set_mcore_fsdp_configs(recipe, comm_overlap_callback_idx, tp_size)

    recipe = set_precision_configs(recipe, compute_dtype, fp8_recipe)
    recipe = set_recompute_configs(recipe, recompute_layers, activation_offload_layers, recompute_modules)

    recipe.trainer.strategy.use_sharp = bool(use_sharp)

    ddp_strategy = recipe.trainer.strategy.ddp
    is_ddp_obj = hasattr(ddp_strategy, "ddp") and not isinstance(ddp_strategy.ddp, str)

    if not is_ddp_obj:
        if use_user_buffer_registration:
            logger.warning("DDP is not configured. Cannot use user buffer registration.")
        return recipe

    # Configure DDP settings (only when DDP object exists)
    ddp_strategy.check_for_nan_in_grad = False
    ddp_strategy.check_for_large_grads = False

    # Configure NCCL User Buffer only when explicitly requested
    if use_user_buffer_registration:
        try:
            if hasattr(ddp_strategy, 'nccl_ub'):
                ddp_strategy.nccl_ub = True
            if hasattr(ddp_strategy, 'fsdp_double_buffer'):
                recipe.trainer.strategy.ddp.fsdp_double_buffer = bool(use_fsdp_double_buffer)
            if hasattr(ddp_strategy, 'keep_fp8_transpose_cache_when_using_custom_fsdp'):
                recipe.trainer.strategy.ddp.keep_fp8_transpose_cache_when_using_custom_fsdp = bool(
                    keep_fsdp_fp8_transpose_cache
                )
                logger.info("NCCL User Buffer registration enabled successfully")
            else:
                logger.warning("NCCL User Buffer property not available in DDP strategy")
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"Failed to enable NCCL User Buffer registration: {e}")
            logger.warning("Continuing without NCCL User Buffer optimization")

    return recipe


def set_primary_perf_configs(
    recipe,
    task: str,
    num_nodes: int,
    num_gpus_per_node: int,
    mbs: int,
    gbs: int,
    max_steps: int,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    vp_size: int,
    ep_size: int,
    etp_size: Optional[int] = None,
    enable_cuda_graphs: bool = False,
    use_mcore_fsdp: bool = False,
    use_fsdp_double_buffer: Optional[bool] = None,
    use_user_buffer_registration: Optional[bool] = None,
    use_sharp: Optional[bool] = None,
    recompute_layers: int = 0,
    activation_offload_layers: int = 0,
    compute_dtype: str = None,
    fp8_recipe: str = None,
    recompute_modules: Optional[List[str]] = None,
    nccl_communicator_config_path: str = None,
    keep_fsdp_fp8_transpose_cache: Optional[bool] = None,
    model_name: str = None,
):
    """Set experiment configs we usually tune for performance of all models."""
    logger.info(f"Setting primary performance configs for {task}")
    # nemo.lightning.Trainer configs
    recipe.trainer.num_nodes = num_nodes
    recipe.trainer.devices = num_gpus_per_node
    recipe.trainer.max_steps = max_steps

    recipe.trainer.val_check_interval = max_steps
    recipe.trainer.limit_val_batches = 0

    # lightning.pytorch.LightningDataModule configs
    recipe.data.micro_batch_size = mbs
    recipe.data.global_batch_size = gbs

    # parallelism configs
    recipe.trainer.strategy.tensor_model_parallel_size = tp_size
    recipe.trainer.strategy.pipeline_model_parallel_size = pp_size
    recipe.trainer.strategy.context_parallel_size = cp_size
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size = None if vp_size == 1 else vp_size
    recipe.trainer.strategy.expert_model_parallel_size = ep_size
    recipe.trainer.strategy.expert_tensor_parallel_size = etp_size
    recipe.trainer.strategy.sequence_parallel = bool(tp_size > 1)

    if nccl_communicator_config_path is not None:
        recipe.trainer.strategy.nccl_communicator_config_path = nccl_communicator_config_path

    # callback configs
    comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
    dp_size = (num_nodes * num_gpus_per_node) / (tp_size * pp_size * cp_size)

    if comm_overlap_callback_idx is not None:
        # WARNING: If True, checkpointing (if enabled) might not work
        recipe.trainer.callbacks[comm_overlap_callback_idx].overlap_param_gather_with_optimizer_step = bool(
            dp_size > 1 and pp_size > 1 and vp_size and vp_size > 1
        )

    # Auto-detect NCCL User Buffer setting based on model size if model_name is provided
    if model_name and use_user_buffer_registration is None:
        MODEL_SIZE_PATTERNS = [
            r'Config(\d+)B',  # Nemotron3Config4B -> 4B
            r'_(\d+)b_',  # _4b_, _70b_, etc.
            r'_(\d+)B_',  # _4B_, _70B_, etc.
            r'(\d+)b_',  # 4b_, 70b_ at start
            r'(\d+)B_',  # 4B_, 70B_ at start
            r'_(\d+)b\d',  # _4b8, _70b8 (followed by digit)
            r'_(\d+)B\d',  # _4B8, _70B8 (followed by digit)
            r'(\d+)B',  # General pattern for XB
            r'(\d+)b',  # lowercase b
        ]
        
        model_size_b = 0
        for pattern in MODEL_SIZE_PATTERNS:
            match = re.search(pattern, model_name, re.IGNORECASE)
            if match:
                model_size_b = int(match.group(1))
                break

        # Enable NCCL User Buffer for models >= 70B parameters
        # Smaller models may not have enough memory headroom for the additional buffer allocation
        use_user_buffer_registration = model_size_b >= 70

        logger.info(f"Model size: {model_size_b}B parameters")
        logger.info(f"NCCL User Buffer auto-enabled: {use_user_buffer_registration} (threshold: 70B+)")

    recipe = set_perf_optimization_configs(
        recipe=recipe,
        use_mcore_fsdp=use_mcore_fsdp,
        enable_cuda_graphs=enable_cuda_graphs,
        task=task,
        tp_size=tp_size,
        compute_dtype=compute_dtype,
        fp8_recipe=fp8_recipe,
        recompute_layers=recompute_layers,
        activation_offload_layers=activation_offload_layers,
        recompute_modules=recompute_modules,
        use_fsdp_double_buffer=use_fsdp_double_buffer,
        use_user_buffer_registration=use_user_buffer_registration,
        use_sharp=use_sharp,
        keep_fsdp_fp8_transpose_cache=keep_fsdp_fp8_transpose_cache,
    )

    return recipe
