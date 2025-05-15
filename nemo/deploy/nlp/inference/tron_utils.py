import datetime
import os
from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional, Union

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.transformer.module import Float16Module, MegatronModule

from nemo.collections.llm.gpt.model.base import GPTConfig
from nemo.collections.llm.t5.model.t5 import T5Config


@dataclass
class RNGConfig:
    """Configuration settings for random number generation."""

    seed: int = 1234
    """Random seed used for python, numpy, pytorch, and cuda."""

    te_rng_tracker: bool = False
    """Use the Transformer Engine version of the random number generator.
    Required for CUDA graphs support."""

    inference_rng_tracker: bool = False
    """Use a random number generator configured for inference."""

    data_parallel_random_init: bool = False
    """Enable random initialization of params across data parallel ranks"""


@dataclass
class DistributedInitConfig:
    """Configuration settings for distributed training initialization."""

    # ---------------- Distributed config. ----------------

    distributed_backend: Literal["nccl", "gloo"] = "nccl"
    """Which backend to use for distributed training."""

    distributed_timeout_minutes: int = 10
    """Timeout minutes for torch.distributed."""

    align_grad_reduce: bool = True
    """If not set, all PP stages will launch gradient reduces simultaneously.
    Otherwise, each PP stage will independently launch as needed.
    """

    local_rank: int = field(default_factory=lambda: int(os.getenv("LOCAL_RANK", "0")))
    """local rank passed from distributed launcher."""

    lazy_mpu_init: bool = False
    """If set to True, initialize_megatron() skips DDP initialization and returns function to
    complete it instead. Also turns on --use-cpu-initialization flag. This is for external DDP
    manager."""

    use_torch_fsdp2: bool = False
    """Use the torch FSDP2 implementation. FSDP2 is not currently working with Pipeline Parallel.
    It is still not in a stable release stage, and may therefore contain bugs or other
    potential issues."""

    nccl_communicator_config_path: Optional[str] = None
    """Path to the yaml file with NCCL communicator configurations. The number of min/max thread
    groups and thread group cluster size of each communicator can be configured by setting
    `min_ctas`, `max_ctas`, and `cga_cluster_size`."""

    use_tp_pp_dp_mapping: bool = False
    """If set, distributed ranks initialize order is changed from tp-dp-pp to tp-pp-dp.
    Make sure EP and CP aren't used with this option enabled.
    """

    use_gloo_process_groups: bool = True
    """If set, create Gloo process groups for communications."""


def get_rank_safe() -> int:
    """Get the rank from torch.distributed or environment variable.

    Returns:
        int: The global rank of the current process.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return int(os.getenv("RANK", "0"))


def get_world_size_safe() -> int:
    """Get the world size from torch.distributed or environment variable.

    Returns:
        int: The total number of processes in the distributed setup.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return int(os.getenv("WORLD_SIZE", "1"))


def get_local_rank_preinit() -> int:
    """Get the local rank from the environment variable, intended for use before full init.

    Returns:
        int: The local rank of the current process.
    """
    return int(os.getenv("LOCAL_RANK", "0"))


def print_rank_0(message: str) -> None:
    """Print a message only on global rank 0.

    Args:
        message (str): The message string to print.
    """
    rank = get_rank_safe()
    if rank == 0:
        print(message, flush=True)


def initialize_distributed(
    model_config: Union[GPTConfig, T5Config],
    dist_config: DistributedInitConfig,
    num_distributed_optimizer_instances: int,
    get_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]],
    get_position_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]],
) -> None:
    """Initialize torch.distributed and core model parallel.

    Args:
        model_config (Union[GPTConfig, T5Config]): Configuration for the model architecture
        dist_config (DistributedInitConfig): Configuration for distributed initialization
        num_distributed_optimizer_instances (int): Number of optimizer instances for distributed training
        get_embedding_ranks (Optional[Callable[[List[int], Optional[int]], List[int]]]): Function to get the ranks for embedding parallel
        get_position_embedding_ranks (Optional[Callable[[List[int], Optional[int]], List[int]]]): Function to get the ranks for position embedding parallel
    """

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        if get_rank_safe() == 0:
            print(
                "torch distributed is already initialized, skipping initialization ...",
                flush=True,
            )
    else:
        if get_rank_safe() == 0:
            print("> initializing torch distributed ...", flush=True)
        else:
            print(f"!!! Current rank: {get_rank_safe()}")

        # Manually set the device ids.
        if device_count > 0:
            torch.cuda.set_device(get_local_rank_preinit())

        # Call the init process
        init_process_group_kwargs = {
            "backend": dist_config.distributed_backend,
            "world_size": get_world_size_safe(),
            "rank": get_rank_safe(),
            "timeout": datetime.timedelta(minutes=dist_config.distributed_timeout_minutes),
        }

        torch.distributed.init_process_group(**init_process_group_kwargs)
        torch.distributed.barrier(device_ids=[get_local_rank_preinit()])

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if parallel_state.model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            parallel_state.initialize_model_parallel(
                model_config.tensor_model_parallel_size,
                model_config.pipeline_model_parallel_size,
                model_config.virtual_pipeline_model_parallel_size,
                model_config.pipeline_model_parallel_split_rank,
                context_parallel_size=model_config.context_parallel_size,
                hierarchical_context_parallel_sizes=model_config.hierarchical_context_parallel_sizes,
                expert_model_parallel_size=model_config.expert_model_parallel_size,
                num_distributed_optimizer_instances=num_distributed_optimizer_instances,
                expert_tensor_parallel_size=model_config.expert_tensor_parallel_size,
                distributed_timeout_minutes=dist_config.distributed_timeout_minutes,
                nccl_communicator_config_path=dist_config.nccl_communicator_config_path,
                order="tp-cp-ep-dp-pp" if not dist_config.use_tp_pp_dp_mapping else "tp-pp-dp",
                encoder_tensor_model_parallel_size=getattr(model_config, "encoder_tensor_model_parallel_size", 0),
                encoder_pipeline_model_parallel_size=getattr(model_config, "encoder_pipeline_model_parallel_size", 0),
                get_embedding_ranks=get_embedding_ranks,
                get_position_embedding_ranks=get_position_embedding_ranks,
                create_gloo_process_groups=dist_config.use_gloo_process_groups,
            )
            if get_rank_safe() == 0:
                print(
                    f"> initialized tensor model parallel with size "
                    f"{parallel_state.get_tensor_model_parallel_world_size()}"
                )
                print(
                    f"> initialized pipeline model parallel with size "
                    f"{parallel_state.get_pipeline_model_parallel_world_size()}"
                )


def _set_random_seed(
    seed_: int,
    data_parallel_random_init: bool = False,
    te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
) -> None:
    """Set random seed for reproducability.

    Args:
        seed_ (int): Base random seed to use
        data_parallel_random_init (bool, optional): Whether to use different seeds for different data parallel ranks. Defaults to False.
        te_rng_tracker (bool, optional): Whether to use Transformer Engine random number generator. Defaults to False.
        inference_rng_tracker (bool, optional): Whether to use a random number generator configured for inference. Defaults to False.
    """
    assert seed_ is not None and seed_ > 0, f"Seed ({seed_}) should be a positive integer."

    import random

    import numpy as np

    # Ensure that different pipeline MP stages get different seeds.
    seed = seed_ + (100 * parallel_state.get_pipeline_model_parallel_rank())
    # Ensure different data parallel ranks get different seeds
    if data_parallel_random_init:
        seed = seed + (10 * parallel_state.get_data_parallel_rank())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        tensor_parallel.model_parallel_cuda_manual_seed(seed, te_rng_tracker, inference_rng_tracker)


def _initialize_tp_communicators(model_config: Union[GPTConfig, T5Config], micro_batch_size: int) -> None:
    """Initialize communicators with user buffers for high-performance tensor-model-parallel communication overlap.

    Args:
        model_config (Union[GPTConfig, T5Config]): Configuration for the model architecture
        micro_batch_size (int): Size of the micro batch
    """
    try:
        import transformer_engine  # noqa: F401
        import yaml
        from transformer_engine.pytorch import module as te_module

    except ImportError:
        raise RuntimeError(
            "Tensor Parallel Communication/GEMM Overlap optimization needs 'yaml' and 'transformer_engine' packages"
        )

    if model_config.tp_comm_overlap_cfg is not None:
        with open(model_config.tp_comm_overlap_cfg, "r") as stream:
            ub_cfgs = yaml.safe_load(stream)
    else:
        ub_cfgs = {}

    input_shape = [
        (model_config.seq_length * micro_batch_size) // model_config.context_parallel_size,
        model_config.hidden_size,
    ]

    # Create required process groups
    bootstrap_backend = getattr(model_config, "tp_comm_bootstrap_backend", "mpi")

    # Set up the appropriate process group and initialize user buffers
    try:
        # Try to use TE's version-specific API if available
        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=model_config.tensor_model_parallel_size,
            use_fp8=(getattr(model_config, "fp8", None) is not None),
            ub_cfgs=ub_cfgs,
            bootstrap_backend=bootstrap_backend,
        )
    except TypeError:
        # Fallback for older TE versions
        if bootstrap_backend != "mpi":
            print(f"Warning: Transformer Engine may only support MPI bootstrap backend")

        # Create a MPI process group for TP communication bootstrap
        torch.distributed.new_group(backend="mpi")

        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=model_config.tensor_model_parallel_size,
            use_fp8=(getattr(model_config, "fp8", None) is not None),
            ub_cfgs=ub_cfgs,
        )


def _get_model_type(model_config: Union[GPTConfig, T5Config]) -> ModelType:
    """Determine the model type from the model configuration.

    Args:
        model_config (Union[GPTConfig, T5Config]): The model configuration object

    Returns:
        ModelType: The model type enum value (encoder_and_decoder or encoder_or_decoder)
    """
    return ModelType.encoder_and_decoder if isinstance(model_config, T5Config) else ModelType.encoder_or_decoder


def get_model_from_config(
    model_config: Union[GPTConfig, T5Config],
    ddp_config: DistributedDataParallelConfig,
    overlap_param_gather_with_optimizer_step: bool = False,
    wrap_with_ddp: bool = True,
    data_parallel_random_init: bool = True,
) -> List[MegatronModule]:
    """Get a model from the given configuration.

    This method should only be called after `init_distributed()`.

    Args:
        model_config (Union[GPTConfig, T5Config]): The model configuration
        ddp_config (DistributedDataParallelConfig): The distributed data parallel configuration
        overlap_param_gather_with_optimizer_step (bool, optional): Whether to overlap parameter gathering with optimizer step. Defaults to False.
        wrap_with_ddp (bool, optional): Whether to wrap the model with DistributedDataParallel. Defaults to True.
        data_parallel_random_init (bool, optional): Whether to initialize data parallel ranks with random seeds. Defaults to True.

    Returns:
        List[MegatronModule]: List of model modules, potentially wrapped with DistributedDataParallel
    """
    model_type = _get_model_type(model_config)
    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
    ):
        assert (
            model_type != ModelType.encoder_and_decoder
        ), "Interleaved schedule not supported for model with both encoder and decoder"
        model = []
        for i in range(parallel_state.get_virtual_pipeline_model_parallel_world_size()):
            parallel_state.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = parallel_state.is_pipeline_first_stage()
            post_process = parallel_state.is_pipeline_last_stage()
            this_model = model_config.configure_model(
                tokenizer=None,
                pre_process=pre_process,
                post_process=post_process,
            )
            this_model.model_type = model_type
            model.append(this_model)
    else:
        pre_process = parallel_state.is_pipeline_first_stage()
        post_process = parallel_state.is_pipeline_last_stage()
        if model_type == ModelType.encoder_and_decoder:
            assert isinstance(model_config, T5Config)
            if parallel_state.get_pipeline_model_parallel_world_size() > 1:
                rank = parallel_state.get_pipeline_model_parallel_rank()
                first_decoder_rank = parallel_state.get_pipeline_model_parallel_decoder_start()
                world_size = parallel_state.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == first_decoder_rank
                post_process = (rank == (first_decoder_rank - 1)) or (rank == (world_size - 1))
            model = model_config.configure_model(
                tokenizer=None,
            )
        else:
            model = model_config.configure_model(
                tokenizer=None,
                pre_process=pre_process,
                post_process=post_process,
            )
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if parallel_state.get_data_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) model parallel rank ({}, {}): {}".format(
                parallel_state.get_tensor_model_parallel_rank(),
                parallel_state.get_pipeline_model_parallel_rank(),
                sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model]),
            ),
            flush=True,
        )

    # GPU allocation.
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if model_config.fp16 or model_config.bf16:
        model = [Float16Module(model_config, model_module) for model_module in model]

    # Handle FP8 parameters if present
    for model_module in model:
        for param in model_module.parameters():
            if hasattr(param, "_fp8_meta") and param._fp8_meta is not None:
                fp8_meta = param._fp8_meta["scaling_fwd"]
                fp8_meta_index = param._fp8_meta_index
                if hasattr(param, "get_high_precision_init_val"):
                    fp8_meta.amax_history[0][fp8_meta_index].copy_(param.get_high_precision_init_val().abs().max())
                else:
                    fp8_meta.amax_history[0][fp8_meta_index] = 0

    if wrap_with_ddp:
        model = [
            DistributedDataParallel(
                config=model_config,
                ddp_config=ddp_config,
                module=model_chunk,
                # Turn off bucketing for model_chunk 2 onwards, since communication for these
                # model chunks is overlapped with compute anyway.
                disable_bucketing=(model_chunk_idx > 0) or overlap_param_gather_with_optimizer_step,
            )
            for (model_chunk_idx, model_chunk) in enumerate(model)
        ]

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()
    return model
