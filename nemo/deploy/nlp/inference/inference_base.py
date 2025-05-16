import atexit
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import megatron.core.dist_checkpointing.serialization as dist_ckpt
import torch
from megatron.core.dist_checkpointing.core import check_is_distributed_checkpoint
from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
from megatron.core.inference.engines.mcore_engine import MCoreEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from megatron.core.inference.text_generation_controllers.text_generation_controller import TextGenerationController
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import MegatronModule

from nemo.collections.llm.gpt.model.base import GPTConfig
from nemo.collections.llm.inference.base import MCoreTokenizerWrappper
from nemo.collections.llm.modelopt import set_modelopt_spec_if_exists_in_ckpt
from nemo.collections.llm.t5.model.t5 import T5Config
from nemo.lightning import io
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.lightning.io.pl import ckpt_to_weights_subdir

from .tron_utils import (
    DistributedInitConfig,
    RNGConfig,
    _initialize_tp_communicators,
    _set_random_seed,
    get_model_from_config,
    get_world_size_safe,
    initialize_distributed,
)

LOGGER = logging.getLogger("NeMo")


def _load_dist_shards_into_model(model: List[MegatronModule], weights_dir: Path) -> None:
    """
    Load a NeMo-2 distributed checkpoint (torch_dist .distcp shards) into
    an already-constructed Megatron model list.

    Args:
        model (List[MegatronModule]): The list of Megatron model modules
        weights_dir (Path): Path to the weights directory containing shards
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


def cleanup_distributed() -> None:
    """
    Clean up the distributed environment by destroying the process group.
    This prevents resource leaks and warnings about destroy_process_group() not being called.
    """
    if torch.distributed.is_initialized():
        LOGGER.info("Cleaning up distributed environment")
        torch.distributed.destroy_process_group()


# Register cleanup function to be called at program exit
atexit.register(cleanup_distributed)


def initialize_megatron_for_inference(
    model_config: Union[GPTConfig, T5Config],
    dist_config: DistributedInitConfig,
    rng_config: RNGConfig,
    micro_batch_size: int,
) -> None:
    """
    Initialize the Megatron-Tron runtime components required for inference.

    Args:
        model_config (Union[GPTConfig, T5Config]): The model configuration object that
                                                  specifies tensor/pipeline parallel sizes
                                                  and model architecture details
        dist_config (DistributedInitConfig): Distributed launcher configuration that controls
                                            torch.distributed process groups
        rng_config (RNGConfig): Configuration for random number generation behavior and seed
        micro_batch_size (int): The micro batch size used during model execution
    """
    initialize_distributed(
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
    Recursively unwrap a wrapped torch.nn.Module and return the underlying module.

    Args:
        m (torch.nn.Module): The (possibly wrapped) PyTorch module

    Returns:
        torch.nn.Module: The innermost unwrapped module
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

    Args:
        model (List[MegatronModule]): Tron model modules list (from get_model_from_config)
        path (Path): Path to NeMo checkpoint directory
    """
    weights_dir = ckpt_to_weights_subdir(path, is_saving=False)
    LOGGER.info(f"Loading NeMo checkpoint from {weights_dir}")

    _load_dist_shards_into_model(model, weights_dir)


def setup_model_and_tokenizer_for_inference(
    checkpoint_path: Union[str, Path],
    tensor_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: Optional[int] = None,
    expert_model_parallel_size: Optional[int] = None,
    params_dtype: Optional[torch.dtype] = None,
    micro_batch_size: Optional[int] = None,
    enable_flash_decode: bool = False,
    enable_cuda_graphs: bool = False,
) -> Tuple[List[MegatronModule], MCoreTokenizerWrappper]:
    """
    Initialize a Megatron-Core model and tokenizer for inference from a NeMo-2.0 checkpoint.

    Args:
        checkpoint_path (Union[str, Path]): Path to the NeMo checkpoint directory
        tensor_model_parallel_size (Optional[int]): Desired tensor-parallel world size (defaults to checkpoint value)
        pipeline_model_parallel_size (Optional[int]): Desired pipeline-parallel world size (defaults to checkpoint value)
        context_parallel_size (Optional[int]): Desired context-parallel world size (defaults to checkpoint value)
        expert_model_parallel_size (Optional[int]): Desired expert parallel world size (defaults to checkpoint value)
        params_dtype (Optional[torch.dtype]): Data type for model parameters (defaults to checkpoint dtype)
        micro_batch_size (Optional[int]): Micro batch size for model execution (defaults to 1)
        enable_flash_decode (bool): Whether to enable flash attention decoding
        enable_cuda_graphs (bool): Whether to enable CUDA graphs optimization

    Returns:
        Tuple[List[MegatronModule], MCoreTokenizerWrappper]: Tuple containing:
            - List of instantiated Megatron-Core modules
            - Tokenizer wrapper with encode/decode interface

    Raises:
        ValueError: If checkpoint_path is not a valid NeMo-2.0 checkpoint
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

    # Enable flash attention
    if enable_flash_decode:
        model_config.flash_decode = True
        model_config.attention_backend = AttnBackend.flash

    # Enable CUDA graphs
    if enable_cuda_graphs:
        model_config.enable_cuda_graph = True
        model_config.use_te_rng_tracker = True
        model_config.inference_rng_tracker = True

    # Create the model using tron APIs
    model = get_model_from_config(
        model_config,
        ddp_config=dist_config,
        wrap_with_ddp=False,  # No need for DDP for inference
    )

    # Ensure model is configured
    for model_module in model:
        if hasattr(model_module, "configure_model") and callable(model_module.configure_model):
            model_module.configure_model()

    # Load checkpoint weights
    load_nemo_checkpoint_to_tron_model(model, checkpoint_path)

    # Get MCore model
    model = [peel(m) for m in model]

    tokenizer = model_context.tokenizer
    tokenizer_wrapper = MCoreTokenizerWrappper(tokenizer)

    return model, tokenizer_wrapper


class MCoreEngineWithCleanup:
    """
    Wrapper around MCoreEngine that ensures proper cleanup of distributed resources.

    This class delegates all operations to the underlying MCoreEngine while ensuring that
    distributed resources are properly cleaned up when the engine is destroyed.
    """

    def __init__(
        self,
        mcore_engine: MCoreEngine,
        model_inference_wrapper: GPTInferenceWrapper,
        tokenizer: MCoreTokenizerWrappper,
    ):
        """
        Initialize the MCoreEngineWithCleanup.

        Args:
            mcore_engine (MCoreEngine): The underlying MCoreEngine instance
            model_inference_wrapper (GPTInferenceWrapper): The model inference wrapper
            tokenizer (MCoreTokenizerWrappper): The tokenizer wrapper
        """
        self.mcore_engine = mcore_engine
        self.model_inference_wrapper = model_inference_wrapper
        self.tokenizer = tokenizer

    def __del__(self):
        # Ensure cleanup happens when the engine is destroyed
        cleanup_distributed()

    def __getattr__(self, name):
        # Delegate all attribute access to the underlying engine
        return getattr(self.mcore_engine, name)


def create_mcore_engine(
    path: Path,
    num_devices: int = None,
    num_nodes: int = None,
    params_dtype: torch.dtype = torch.bfloat16,
    inference_batch_times_seqlen_threshold: int = 32768,
    inference_max_seq_length: int = 4096,
    max_batch_size: int = 8,
    random_seed: Optional[int] = None,
    tensor_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: Optional[int] = None,
    expert_model_parallel_size: Optional[int] = None,
    enable_flash_decode: bool = False,
    enable_cuda_graphs: bool = False,
) -> Tuple[MCoreEngineWithCleanup, GPTInferenceWrapper, MCoreTokenizerWrappper]:
    """
    Set up the model, tokenizer and MCoreEngine for inference.

    Args:
        path (Path): Path to the checkpoint file
        params_dtype (torch.dtype): Data type for model parameters (default: torch.bfloat16)
        inference_batch_times_seqlen_threshold (int): Threshold for batch size times sequence length
        inference_max_seq_length (int): Maximum sequence length for inference
        max_batch_size (int): Maximum batch size for inference
        random_seed (Optional[int]): Random seed for reproducibility
        tensor_model_parallel_size (Optional[int]): Size of tensor model parallelism
        pipeline_model_parallel_size (Optional[int]): Size of pipeline model parallelism
        context_parallel_size (Optional[int]): Size of context parallelism
        expert_model_parallel_size (Optional[int]): Size of expert model parallelism
        enable_flash_decode (bool): Whether to enable flash attention decoding
        enable_cuda_graphs (bool): Whether to enable CUDA graphs optimization

    Returns:
        Tuple[MCoreEngineWithCleanup, GPTInferenceWrapper, MCoreTokenizerWrappper]: Tuple containing:
            - MCoreEngineWithCleanup: Engine for text generation with proper cleanup
            - GPTInferenceWrapper: Inference-wrapped model
            - MCoreTokenizerWrappper: Tokenizer wrapper
    """
    # Load model context to get default parallelism settings from checkpoint
    checkpoint_path = Path(path)
    model_context = io.load_context(path=ckpt_to_context_subdir(checkpoint_path), subpath="model")
    model_config = model_context.config

    # Use checkpoint values as defaults if not specified
    tp_size = (
        tensor_model_parallel_size
        if tensor_model_parallel_size is not None
        else model_config.tensor_model_parallel_size
    )
    pp_size = (
        pipeline_model_parallel_size
        if pipeline_model_parallel_size is not None
        else model_config.pipeline_model_parallel_size
    )
    cp_size = context_parallel_size if context_parallel_size is not None else model_config.context_parallel_size
    ep_size = (
        expert_model_parallel_size
        if expert_model_parallel_size is not None
        else model_config.expert_model_parallel_size
    )

    # Default to 1 for any parallelism dimension that's None
    tp_size = tp_size if tp_size is not None else 1
    pp_size = pp_size if pp_size is not None else 1
    cp_size = cp_size if cp_size is not None else 1
    ep_size = ep_size if ep_size is not None else 1

    # Calculate total devices needed for the parallelism configuration
    total_devices_needed = tp_size * pp_size * cp_size * ep_size
    if num_devices is None:
        total_devices_available = get_world_size_safe()
    else:
        total_devices_available = num_nodes * num_devices

    if total_devices_needed > total_devices_available:
        raise ValueError(
            f"Insufficient devices for requested parallelism configuration. "
            f"Need {total_devices_needed} devices "
            f"(TP={tp_size} × PP={pp_size} × CP={cp_size} × EP={ep_size}), "
            f"but only have {total_devices_available} devices "
            f"({num_nodes} nodes × {num_devices} devices)."
        )

    modelList, tokenizer = setup_model_and_tokenizer_for_inference(
        checkpoint_path=path,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        params_dtype=params_dtype,
        enable_flash_decode=enable_flash_decode,
        enable_cuda_graphs=enable_cuda_graphs,
    )
    model = modelList[0]
    vocab_size = None
    if tokenizer is not None:
        vocab_size = tokenizer.vocab_size
    elif hasattr(model.config, "vocab_size"):
        vocab_size = model.config.vocab_size
    else:
        raise ValueError("Unable to find vocab size.")

    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=model.config.hidden_size,
        params_dtype=params_dtype,
        inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
        padded_vocab_size=vocab_size,
        inference_max_seq_length=inference_max_seq_length,
        inference_max_requests=max_batch_size,
    )

    model_inference_wrapper = GPTInferenceWrapper(model, inference_wrapper_config)
    text_generation_controller = TextGenerationController(
        inference_wrapped_model=model_inference_wrapper, tokenizer=tokenizer
    )
    mcore_engine = MCoreEngine(
        text_generation_controller=text_generation_controller, max_batch_size=max_batch_size, random_seed=random_seed
    )

    # Wrap the engine to ensure cleanup
    wrapped_engine = MCoreEngineWithCleanup(mcore_engine, model_inference_wrapper, tokenizer)

    return wrapped_engine, model_inference_wrapper, tokenizer
