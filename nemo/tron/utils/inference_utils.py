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


# This module provides utilities for loading NeMo-2.0 checkpoints using MCore utilities
# for inference purposes.


from pathlib import Path
from typing import List, Optional, Tuple, Union

import megatron.core.dist_checkpointing.serialization as dist_ckpt
import torch
from megatron.core.dist_checkpointing.core import check_is_distributed_checkpoint
from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
from megatron.core.transformer.module import MegatronModule

from nemo.collections.llm.gpt.model.base import GPTConfig
from nemo.collections.llm.inference.base import MCoreTokenizerWrappper
from nemo.collections.llm.modelopt import set_modelopt_spec_if_exists_in_ckpt
from nemo.collections.llm.t5.model.t5 import T5Config
from nemo.lightning import io
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.lightning.io.pl import ckpt_to_weights_subdir
from nemo.tron.config import DistributedInitConfig, RNGConfig
from nemo.tron.init import _initialize_distributed, _initialize_tp_communicators, _set_random_seed
from nemo.tron.model import get_model_from_config
from nemo.tron.utils.common_utils import print_rank_0


def _load_dist_shards_into_model(model: List[MegatronModule], weights_dir: Path):
    """
    Load a NeMo-2 distributed checkpoint (torch_dist .distcp shards) into
    an already-constructed Megatron model list.
    """
    # Build a sharded_state_dict that mirrors `generate_state_dict()`
    sharded_state_dict = {}
    if len(model) == 1:
        sharded_state_dict["model"] = MegatronModule.sharded_state_dict(model[0])
    else:  # virtual pipeline schedule
        for i, m in enumerate(model):
            sharded_state_dict[f"model{i}"] = MegatronModule.sharded_state_dict(m)

    # Get the default strategy for that directory
    load_strategy = get_default_load_sharded_strategy(str(weights_dir))

    # Materialise the shards in-place
    dist_ckpt.load(
        sharded_state_dict,
        str(weights_dir),
        load_strategy,
    )

    # Normal torch `load_state_dict()` still required for non-sharded
    # buffers (pos-embeddings, LayerNorm bias, etc.)
    if len(model) == 1:
        model[0].load_state_dict(sharded_state_dict["model"], strict=False)
    else:
        for i, m in enumerate(model):
            m.load_state_dict(sharded_state_dict[f"model{i}"], strict=False)


def initialize_megatron_for_inference(
    model_config: GPTConfig | T5Config,
    dist_config: DistributedInitConfig,
    rng_config: RNGConfig,
    micro_batch_size: int,
) -> None:
    """
    Initialize the Megatron-Tron runtime components required for *inference*.

    This helper is a lightweight alternative to ``nemo.tron.init.initialize_megatron``.
    It skips all training-specific initialization routines (such as the micro-batch
    calculator and checkpoint handling) and performs only the steps strictly
    necessary for running a model in inference / evaluation mode.

    Args:
        model_config (GPTConfig | T5Config):
            The model configuration object that specifies tensor/pipeline parallel sizes,
            model architecture details, and runtime flags such as
            ``tp_comm_overlap`` (tensor-parallel communication overlap).
        dist_config (DistributedInitConfig):
            Distributed launcher configuration that controls how the global and local
            ``torch.distributed`` process groups are created (backend, timeouts,
            ranks, world size, etc.).
        rng_config (RNGConfig):
            Configuration that governs all random-number-generation behaviour,
            including the global seed, whether each data parallel worker should
            advance the RNG state independently, and which CUDA RNG trackers
            (Transformer Engine vs. inference) should be initialised.
        micro_batch_size (int):
            The micro batch size used during model execution.  Although not
            strictly required for the distributed initialisation itself, it is
            needed later if tensor-parallel communicator overlap is enabled, as
            that feature performs a small warm-up forward pass that depends on the
            batch size.
    """
    _initialize_distributed(
        model_config=model_config,
        dist_config=dist_config,
        num_distributed_optimizer_instances=1,
        get_embedding_ranks=None,
        get_position_embedding_ranks=None,
    )
    _set_random_seed(
        rng_config.seed,
        rng_config.data_parallel_random_init,
        rng_config.te_rng_tracker,
        rng_config.inference_rng_tracker,
    )
    if model_config.tp_comm_overlap:
        _initialize_tp_communicators(model_config, micro_batch_size)


def peel(m: torch.nn.Module) -> torch.nn.Module:
    """
    Recursively unwrap a wrapped ``torch.nn.Module`` and return the underlying
    (innermost) module.

    Wrapper classes such as ``torch.nn.DataParallel`` or
    ``torch.nn.parallel.DistributedDataParallel`` (and various custom
    wrappers) typically expose the wrapped model through a ``module``
    attribute.  This utility walks down the chain of ``module`` attributes
    until a layer that no longer possesses such an attribute is reached.

    Args:
        m (torch.nn.Module): The (possibly wrapped) PyTorch module.

    Returns:
        torch.nn.Module: The deepest unwrapped module.
    """
    while hasattr(m, "module"):
        m = m.module
    return m


def load_nemo_checkpoint_to_tron_model(
    model: List[MegatronModule],
    path: Path,
) -> None:
    """
    Load NeMo checkpoint weights into a Tron model.
    Uses Tron's load_checkpoint for proper handling of distributed checkpoints.

    Args:
        model (list[MegatronModule]): Tron model modules list (from get_model_from_config)
        path (Path): Path to NeMo checkpoint directory
    """
    weights_dir = ckpt_to_weights_subdir(path, is_saving=False)
    print_rank_0(f"Loading NeMo checkpoint from {weights_dir}")

    # TODO: Implement PEFT loading

    _load_dist_shards_into_model(model, weights_dir)


def setup_model_and_tokenizer_for_inference(
    checkpoint_path: Union[str, Path],
    tensor_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: Optional[int] = None,
    expert_model_parallel_size: Optional[int] = None,
    params_dtype: Optional[torch.dtype] = None,
    micro_batch_size: Optional[int] = None,
) -> Tuple[List[MegatronModule], MCoreTokenizerWrappper]:
    """
    Initialise a Megatron-Core (Tron) model **and** its tokenizer for
    forward-only inference from a NeMo-2.0 checkpoint directory.

    Args:
        checkpoint_path (Union[str, Path]): Path to the folder that contains the NeMo
            checkpoint (i.e., the directory with the `.nemo2` context and weight
            sub-directories).
        tensor_model_parallel_size (int, optional): Desired tensor-parallel world size
            for inference. Defaults to the value stored in the checkpoint when ``None``.
        pipeline_model_parallel_size (int, optional): Desired pipeline-parallel world
            size. Defaults to the checkpoint value when ``None``.
        context_parallel_size (int, optional): Desired context-parallel world size.
            Defaults to the checkpoint value when ``None``.
        expert_model_parallel_size (int, optional): Desired expert (MoE) parallel
            world size. Defaults to the checkpoint value when ``None``.
        params_dtype (torch.dtype, optional): Data type (``torch.float16``,
            ``torch.bfloat16``, ``torch.float32``, …) to which model parameters should be
            cast before inference. If ``None`` the dtype present in the checkpoint is
            used.
        micro_batch_size (int, optional): The micro batch size used during model execution.
            Defaults to 1.

    Returns:
        Tuple[List[MegatronModule], MCoreTokenizerWrappper]:
            * **model** – A list containing the instantiated Megatron-Core module(s).
              For virtual pipeline parallelism the list has one entry per virtual
              stage, otherwise a single entry.
            * **tokenizer** – A tokenizer wrapper exposing the standard
              ``encode``/``decode`` interface expected by the Tron runtime.

    Raises:
        ValueError: If *checkpoint_path* does not point to a valid NeMo-2.0
            checkpoint or if an unsupported checkpoint format is encountered.
    """
    checkpoint_path = Path(checkpoint_path)

    # Load model context for config and tokenizer
    model_context = io.load_context(path=ckpt_to_context_subdir(checkpoint_path), subpath="model")

    model_config = model_context.config

    # Apply ModelOpt specs if they exist in the checkpoint
    set_modelopt_spec_if_exists_in_ckpt(model_context, checkpoint_path)

    if tensor_model_parallel_size is not None:
        model_config.tensor_model_parallel_size = tensor_model_parallel_size
    if pipeline_model_parallel_size is not None:
        model_config.pipeline_model_parallel_size = pipeline_model_parallel_size
    if context_parallel_size is not None:
        model_config.context_parallel_size = context_parallel_size
    if expert_model_parallel_size is not None:
        model_config.expert_model_parallel_size = expert_model_parallel_size

    if params_dtype is None:
        params_dtype = model_config.params_dtype

    if micro_batch_size is None:
        micro_batch_size = 1

    is_dist_ckpt = check_is_distributed_checkpoint(ckpt_to_weights_subdir(checkpoint_path, is_saving=False))
    if not is_dist_ckpt:
        raise ValueError("Checkpoint is not a NeMo-2 distributed checkpoint")

    # Initialize Megatron for inference
    rng_config = RNGConfig(inference_rng_tracker=True)
    dist_config = DistributedInitConfig(distributed_backend="nccl")
    initialize_megatron_for_inference(model_config, dist_config, rng_config, micro_batch_size)

    # Needed for model creation
    if not model_config.vocab_size:
        model_config.vocab_size = model_context.tokenizer.vocab_size

    # Create the model using tron APIs
    model = get_model_from_config(
        model_config,
        ddp_config=dist_config,
        wrap_with_ddp=False,  # No need for DDP for inference
    )

    # Ensure model is configured (similar to model.configure_model() in the original)
    for model_module in model:
        if hasattr(model_module, "configure_model") and callable(model_module.configure_model):
            model_module.configure_model()

    # Load checkpoint weights using tron's load_checkpoint
    load_nemo_checkpoint_to_tron_model(model, checkpoint_path)

    # Get MCore model
    model = [peel(m) for m in model]

    tokenizer = model_context.tokenizer
    tokenizer_wrapper = MCoreTokenizerWrappper(tokenizer)

    return model, tokenizer_wrapper


if __name__ == "__main__":
    model, tokenizer = setup_model_and_tokenizer_for_inference(
        checkpoint_path="/opt/checkpoints/hf_llama31_8B_nemo2.nemo",
        params_dtype=torch.bfloat16,
    )
