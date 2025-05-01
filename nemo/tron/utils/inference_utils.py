from pathlib import Path
from typing import List, Optional, Tuple, Union

import megatron.core.dist_checkpointing.serialization as dist_ckpt
import torch
from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.module import MegatronModule

from nemo.collections.llm.inference.base import MCoreTokenizerWrappper
from nemo.collections.llm.modelopt import set_modelopt_spec_if_exists_in_ckpt
from nemo.lightning import io
from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.lightning.io.pl import ckpt_to_weights_subdir
from nemo.tron.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedInitConfig,
    GPTDatasetConfig,
    LoggerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
    RNGConfig,
)
from nemo.tron.init import initialize_megatron
from nemo.tron.model import get_model_from_config
from nemo.tron.state import GlobalState
from nemo.tron.utils.common_utils import get_world_size_safe, print_rank_0


def _load_dist_shards_into_model(model: List[MegatronModule], weights_dir: Path):
    """
    Load a NeMo-2 distributed checkpoint (torch_dist .distcp shards) into
    an already-constructed Megatron model list.
    """
    # 1. Build a sharded_state_dict that mirrors `generate_state_dict()`
    sharded_state_dict = {}
    if len(model) == 1:
        sharded_state_dict["model"] = MegatronModule.sharded_state_dict(model[0])
    else:  # virtual pipeline schedule
        for i, m in enumerate(model):
            sharded_state_dict[f"model{i}"] = MegatronModule.sharded_state_dict(m)

    # 2. Get the default strategy for that directory
    load_strategy = get_default_load_sharded_strategy(str(weights_dir))

    # 3. Materialise the shards in-place
    dist_ckpt.load(
        sharded_state_dict,
        str(weights_dir),
        load_strategy,
    )

    # 4. Normal torch `load_state_dict()` still required for non-sharded
    #    buffers (pos-embeddings, LayerNorm bias, etc.)
    if len(model) == 1:
        model[0].load_state_dict(sharded_state_dict["model"], strict=False)
    else:
        for i, m in enumerate(model):
            m.load_state_dict(sharded_state_dict[f"model{i}"], strict=False)


def is_nemo2_dist_ckpt_without_tracker(weights_dir: Path) -> bool:
    """
    Determine whether the supplied directory looks like a NeMo-2 distributed
    checkpoint that is *missing* its training-state tracker file.

    A typical NeMo-2 distributed checkpoint (saved with ``--format torch_dist``)
    stores model parameters in sharded ``*.distcp`` files inside a rank-specific
    folder (``weights_dir``) and accompanies them with a ``metadata.json`` file.
    During training a top-level tracker file named ``latest_train_state.pt`` is
    also written next to the rank folders.  For inference-only checkpoints this
    tracker might be absent.  This helper lets callers tell those two situations
    apart.

    Args:
        weights_dir (Path): Path to the rank-specific directory that should contain
            sharded weight files produced by NeMo-2 (e.g.
            ``.../checkpoints/global_step123/rank_00``).

    Returns:
        bool: ``True`` if ``weights_dir`` contains a ``metadata.json`` and the parent
            directory does *not* contain ``latest_train_state.pt``; ``False`` otherwise.
    """

    tracker = weights_dir.parent / "latest_train_state.pt"
    meta = weights_dir / "metadata.json"
    return meta.is_file() and not tracker.exists()


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


def is_distributed_checkpoint(path: Path) -> bool:
    """
    Check if a directory contains a distributed checkpoint.

    Args:
        path (Path): Path to check

    Returns:
        bool: True if the directory contains .distcp files, False otherwise
    """
    # Look for .metadata or .distcp files which indicate a distributed checkpoint
    metadata_file = path / ".metadata"
    if metadata_file.exists():
        return True

    # Check for any .distcp files
    for file in path.glob("**/*.distcp"):
        return True

    return False


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

    if is_nemo2_dist_ckpt_without_tracker(weights_dir):
        _load_dist_shards_into_model(model, weights_dir)
        return
    raise ValueError("Checkpoint is not a NeMo-2 distributed checkpoint")


def setup_model_and_tokenizer_for_inference(
    checkpoint_path: Union[str, Path],
    tensor_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: Optional[int] = None,
    expert_model_parallel_size: Optional[int] = None,
    params_dtype: Optional[torch.dtype] = None,
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

    # 1. Load model context for config and tokenizer
    model_context = io.load_context(path=ckpt_to_context_subdir(checkpoint_path), subpath="model")

    # 2. Extract model configuration from the loaded context
    model_config = model_context.config

    # 3. Apply ModelOpt specs if they exist in the checkpoint
    set_modelopt_spec_if_exists_in_ckpt(model_context, checkpoint_path)

    # 4. Update parallelism settings if provided
    if tensor_model_parallel_size is not None:
        model_config.tensor_model_parallel_size = tensor_model_parallel_size
    if pipeline_model_parallel_size is not None:
        model_config.pipeline_model_parallel_size = pipeline_model_parallel_size
    if context_parallel_size is not None:
        model_config.context_parallel_size = context_parallel_size
    if expert_model_parallel_size is not None:
        model_config.expert_model_parallel_size = expert_model_parallel_size

    # 5. Use the model's params_dtype if not specified
    if params_dtype is None:
        params_dtype = model_config.params_dtype

    # 6. Calculate correct data parallel size
    world_size = get_world_size_safe()

    # Ensure non-zero values for parallelism dimensions
    tensor_parallel_size = max(model_config.tensor_model_parallel_size, 1)
    pipeline_parallel_size = max(model_config.pipeline_model_parallel_size, 1)
    context_parallel_size = max(model_config.context_parallel_size, 1)
    expert_parallel_size = max(model_config.expert_model_parallel_size, 1)

    # Calculate data parallel size
    model_parallel_size = tensor_parallel_size * pipeline_parallel_size * context_parallel_size * expert_parallel_size
    data_parallel_size = max(world_size // model_parallel_size, 1)  # Ensure at least 1

    # 7. Setup minimal config container for inference
    micro_batch_size = 1
    global_batch_size = micro_batch_size * data_parallel_size

    # Set checkpoint format to match what we're loading
    is_dist_ckpt = is_distributed_checkpoint(ckpt_to_weights_subdir(checkpoint_path, is_saving=False))
    ckpt_format = "torch_dist" if is_dist_ckpt else "torch"

    # TODO: Create a config container for inference
    cfg = ConfigContainer(
        model_config=model_config,
        train_config=TrainingConfig(
            micro_batch_size=micro_batch_size, global_batch_size=global_batch_size, train_iters=1
        ),
        optimizer_config=OptimizerConfig(lr=0.0, weight_decay=0.0, optimizer="adam"),
        scheduler_config=SchedulerConfig(lr_decay_style="constant"),
        dataset_config=GPTDatasetConfig(
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            sequence_length=model_config.seq_length,
            random_seed=1234,
        ),
        logger_config=LoggerConfig(),
        tokenizer_config=TokenizerConfig(
            tokenizer_type=getattr(model_context.tokenizer, "tokenizer_type", "HuggingFaceTokenizer"),
            tokenizer_model=getattr(model_context.tokenizer, "name", None),
        ),
        checkpoint_config=CheckpointConfig(
            load=str(checkpoint_path),  # Will be updated during loading
            load_optim=False,
            load_rng=False,
            ckpt_format=ckpt_format,
            auto_detect_ckpt_format=True,  # Let it auto-detect the format
        ),
        dist_config=DistributedInitConfig(distributed_backend="nccl"),
        rng_config=RNGConfig(inference_rng_tracker=True),
    )

    # 8. Set data_parallel_size explicitly
    cfg.data_parallel_size = data_parallel_size

    # 9. Initialize Megatron using tron API
    initialize_megatron(cfg=cfg)

    # 10. Create the model using tron APIs
    model = get_model_from_config(
        cfg.model_config,
        ddp_config=cfg.ddp_config,
        wrap_with_ddp=False,  # No need for DDP for inference
        tokenizer=model_context.tokenizer,
    )

    # 11. Ensure model is configured (similar to model.configure_model() in the original)
    for model_module in model:
        if hasattr(model_module, "configure_model") and callable(model_module.configure_model):
            model_module.configure_model()

    # 12. Load checkpoint weights using tron's load_checkpoint
    load_nemo_checkpoint_to_tron_model(model, checkpoint_path)

    # 13. Get MCore model
    model = [peel(m) for m in model]

    # 14. Get tokenizer from the original model context
    tokenizer = model_context.tokenizer
    tokenizer_wrapper = MCoreTokenizerWrappper(tokenizer)

    return model, tokenizer_wrapper


if __name__ == "__main__":
    model, tokenizer = setup_model_and_tokenizer_for_inference(
        checkpoint_path="/opt/checkpoints/hf_llama31_8B_nemo2.nemo",
        params_dtype=torch.bfloat16,
    )
