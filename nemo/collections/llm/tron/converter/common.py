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

import logging
import os
import socket
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Optional

import torch
import torch.distributed as dist
import yaml
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.strategies.torch import TorchDistLoadShardedStrategy
from megatron.core.optimizer import OptimizerConfig
from megatron.core.transformer.module import MegatronModule

from nemo.collections.llm.gpt.model.base import GPTConfig, torch_dtype_from_mcore_config
from nemo.collections.llm.t5.model.t5 import T5Config
from nemo.collections.llm.tron.checkpointing import save_checkpoint
from nemo.collections.llm.tron.config import CheckpointConfig, ConfigContainer, LoggerConfig, TokenizerConfig
from nemo.collections.llm.tron.state import GlobalState
from nemo.collections.llm.tron.tokenizers.tokenizer import _HuggingFaceTokenizer
from nemo.collections.llm.tron.utils.instantiate_utils import instantiate

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig


logger = logging.getLogger(__name__)

HF_ASSETS_DIR = "hf_assets"


@contextmanager
def megatron_cpu_init_context(config: Any) -> Generator[None, None, None]:
    """Context manager to temporarily force CPU initialization for Megatron models.

    This is useful when initializing a model on a system without GPUs or when
    memory constraints prevent GPU initialization.

    Args:
        config: The Megatron model configuration object (e.g., GPTConfig).
            Must have a `use_cpu_initialization` attribute.

    Yields:
        None. The context modifies the config in place.
    """
    _orig_use_cpu_initialization = config.use_cpu_initialization

    config.use_cpu_initialization = True

    yield

    config.use_cpu_initialization = _orig_use_cpu_initialization


@contextmanager
def temporary_distributed_context() -> Generator[None, None, None]:
    """Context manager to temporarily initialize a minimal distributed environment.

    Sets up a single-process Gloo backend, initializes Megatron model parallel state,
    yields control, and then cleans up the distributed environment.
    Useful for operations that require Megatron's parallel state but should run
    standalone (e.g., loading distributed checkpoints).

    Yields:
        None.
    """
    if "MASTER_ADDR" in os.environ and "MASTER_PORT" in os.environ:
        init_method = None
    else:
        # Find an available port dynamically
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            addr, port = s.getsockname()

        init_method = f"tcp://{addr}:{port}"

    dist.init_process_group(backend="gloo", init_method=init_method, world_size=1, rank=0)
    parallel_state.initialize_model_parallel()
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def get_full_mcore_state_dict(dist_ckpt_folder: Path, model_cfg: Any) -> dict[str, torch.Tensor]:
    """Load a full, unsharded state dict from a NeMo distributed checkpoint.

    Initializes a minimal distributed environment and a CPU model instance
    to load the sharded state dict using the TorchDistLoadShardedStrategy.

    Args:
        dist_ckpt_folder: Path to the NeMo distributed checkpoint directory
                          (e.g., /path/to/model/checkpoints/iter_0000001).
        model_cfg: The Megatron model configuration object (e.g., GPTConfig)
                   corresponding to the checkpoint.

    Returns:
        A dictionary containing the full, unsharded model state_dict.
    """
    with temporary_distributed_context():
        if model_cfg.params_dtype != torch_dtype_from_mcore_config(model_cfg):
            logger.info(
                f"Converting params_dtype from {model_cfg.params_dtype} to {torch_dtype_from_mcore_config(model_cfg)}"
            )
            model_cfg.params_dtype = torch_dtype_from_mcore_config(model_cfg)

        with megatron_cpu_init_context(model_cfg):
            model = model_cfg.configure_model(None)

        strategy = TorchDistLoadShardedStrategy()
        state_dict = strategy.load(model.sharded_state_dict(), Path(dist_ckpt_folder))
        del model

    return state_dict


def save_hf_tokenizer_assets(tokenizer_name_or_path: str, save_path: str = "/tmp/nemo_tokenizer") -> str:
    """Download and save tokenizer assets from Hugging Face Hub or a local path.

    Uses `transformers.AutoTokenizer` to load and then save the tokenizer files.

    Args:
        tokenizer_name_or_path: The name or path of the Hugging Face tokenizer.
        save_path: The directory where the tokenizer assets will be saved.

    Returns:
        The path where the assets were saved.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    # Save tokenizer assets to save_path.
    tok.save_pretrained(save_path)
    return save_path


def dtype_from_str(dtype: str) -> torch.dtype:
    """Convert a string representation of a dtype to a torch.dtype.

    Handles common variations like 'fp16', 'bf16-mixed'. Defaults to float32
    for unrecognized strings.

    Args:
        dtype: The string representation (e.g., "bf16", "fp16", "float32").

    Returns:
        The corresponding torch.dtype.
    """
    assert isinstance(dtype, str)
    if dtype in ("float16", "fp16", "16", "16-mixed"):
        return torch.float16
    elif dtype in ("bfloat16", "bf16-mixed"):
        return torch.bfloat16
    else:
        return torch.float32


def dtype_from_hf(config: Any) -> torch.dtype:
    """Extract the torch.dtype from a Hugging Face PretrainedConfig object.

    Args:
        config: A Hugging Face model config object (must have `torch_dtype` attribute).

    Returns:
        The corresponding torch.dtype.

    Raises:
        ValueError: If the `torch_dtype` attribute is not a recognized string or torch.dtype.
        AttributeError: If the config object does not have a `torch_dtype` attribute.
    """
    assert hasattr(config, "torch_dtype"), "Expected config to have attr `torch_dtype`"
    torch_dtype = config.torch_dtype
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    elif isinstance(torch_dtype, str):
        return dtype_from_str(torch_dtype)
    else:
        raise ValueError("torch_dtype is not of type str/torch.dtype")


class _ModelState:
    """Helper class for used to modify state dict of a source model during model conversion."""

    def __init__(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Initializes the _ModelState object.

        Args:
            state_dict: The initial state dictionary to wrap.
        """
        self._state_dict = state_dict

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Returns the underlying state dictionary.

        Returns:
            The state dictionary managed by this object.
        """
        return self._state_dict

    def to(self, dtype: torch.dtype) -> None:
        """Converts all tensors in the state dictionary to the specified dtype in-place.

        Logs a warning if any tensor's dtype is actually changed.

        Args:
            dtype: The target torch.dtype to convert tensors to.
        """
        for k, v in self._state_dict.items():
            if v.dtype != dtype:
                logger.warning(f"Converting {k} from {v.dtype} (source model) to {dtype} (target model)")
            self._state_dict[k] = v.to(dtype)


class BaseImporter(ABC):
    """Abstract Base Class for importing models from Hugging Face format to NeMo Tron.

    Provides a common structure and utilities for conversion.
    Subclasses must implement model initialization (`init_hf_model`),
    config mapping (`hf_config`, `tron_config`), and state dict conversion
    (`convert_state`).

    Args:
        input_path: Path to the input Hugging Face model directory.
        output_path: Path to the directory where the NeMo checkpoint will be saved.
    """

    def __init__(self, input_path: str | Path, output_path: str | Path) -> None:
        self.input_path = Path(input_path) if isinstance(input_path, str) else input_path
        self.output_path = Path(output_path) if isinstance(output_path, str) else output_path
        (self.output_path / HF_ASSETS_DIR).mkdir(parents=True, exist_ok=True)
        self._hf_config = None
        self._tron_config = None
        self.hf_config.save_pretrained(str(self.output_path / HF_ASSETS_DIR))

    @property
    def tokenizer(self) -> "_HuggingFaceTokenizer":
        """Get a NeMo _HuggingFaceTokenizer wrapper for the source HF model.

        Downloads/copies the tokenizer assets and initializes the wrapper.

        Returns:
            A NeMo _HuggingFaceTokenizer instance.
        """
        return _HuggingFaceTokenizer(
            save_hf_tokenizer_assets(str(self.input_path), str(self.output_path / HF_ASSETS_DIR))
        )

    def init_tron_model(self, cfg: GPTConfig | T5Config) -> list[MegatronModule]:
        """Initialize the target NeMo Tron model on CPU.

        Args:
            cfg: The NeMo Tron model configuration (e.g., GPTConfig).

        Returns:
            A list containing the initialized NeMo Tron model module.
        """
        with megatron_cpu_init_context(cfg):
            model = cfg.configure_model(tokenizer=self.tokenizer)
        return [model]

    @abstractmethod
    def init_hf_model(self) -> Any:
        """Initialize the source Hugging Face model.

        Must be implemented by subclasses.

        Returns:
            The initialized Hugging Face model instance.
        """
        raise NotImplementedError

    @abstractmethod
    def convert_state(self, source: Any, target: Any) -> None:
        """Convert the state dict from the source HF model to the target NeMo model.

        Must be implemented by subclasses.

        Args:
            source: The source Hugging Face model instance.
            target: The target NeMo Tron model instance.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def hf_config(self) -> "PretrainedConfig":
        """Get the Hugging Face configuration object for the source model.

        Must be implemented by subclasses.

        Returns:
            A Hugging Face PretrainedConfig instance.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def tron_config(self) -> GPTConfig | T5Config:
        """Get the NeMo Tron configuration object derived from the HF config.

        Must be implemented by subclasses.

        Returns:
            A NeMo Tron configuration instance (e.g., GPTConfig).
        """
        raise NotImplementedError

    def apply(self) -> Path:
        """Run the full conversion process from Hugging Face to NeMo Tron.

        Initializes source and target models, converts the state dict,
        and saves the result as a NeMo distributed checkpoint.

        Returns:
            Path to the saved NeMo Tron checkpoint directory.
        """
        source = self.init_hf_model()

        with temporary_distributed_context():
            target = self.init_tron_model(self.tron_config)
            self.convert_state(source, target[0])
            state = GlobalState()
            state.cfg = ConfigContainer(
                model_config=self.tron_config,
                train_config=None,
                optimizer_config=OptimizerConfig(use_distributed_optimizer=False),
                ddp_config=None,
                scheduler_config=None,
                dataset_config=None,
                logger_config=LoggerConfig(),
                tokenizer_config=None,
                checkpoint_config=CheckpointConfig(
                    async_save=False, save=str(self.output_path), save_optim=False, ckpt_format="torch_dist"
                ),
                dist_config=None,
            )
            save_checkpoint(
                state=state,
                model=target,
                optimizer=None,
                opt_param_scheduler=None,
                num_floating_point_operations_so_far=0,
            )

        print(f"Converted {self.input_path} to {self.output_path} in {source.dtype}.")

        return self.output_path


class BaseExporter(ABC):
    """Abstract Base Class for exporting models from NeMo Tron format to Hugging Face.

    Provides a common structure and utilities for conversion.
    Subclasses must implement config mapping (`hf_config`) and state dict
    conversion (`convert_state`).

    Args:
        input_path: Path to the input NeMo Tron checkpoint directory.
        output_path: Path to the directory where the Hugging Face model will be saved.
        hf_tokenizer_path: Optional path or name to a Hugging Face tokenizer to include
                           with the exported model. If None, tries to find one in the
                           NeMo checkpoint.
    """

    def __init__(
        self, input_path: str | Path, output_path: str | Path, hf_tokenizer_path: Optional[str] = None
    ) -> None:
        self.input_path = Path(input_path) if isinstance(input_path, str) else input_path
        self.output_path = Path(output_path) if isinstance(output_path, str) else output_path
        self._hf_config = None
        self._tron_config = None
        self._hf_tokenizer_path = hf_tokenizer_path
        self._tokenizer = None

    @property
    @abstractmethod
    def hf_config(self) -> "PretrainedConfig":
        """Get the Hugging Face configuration object derived from the NeMo config.

        Must be implemented by subclasses.

        Returns:
            A Hugging Face PretrainedConfig instance.
        """
        raise NotImplementedError

    @property
    def tron_config(self) -> GPTConfig | T5Config:
        """Get the NeMo Tron configuration loaded from the checkpoint.

        Returns:
            The loaded NeMo Tron configuration instance (e.g., GPTConfig).

        Raises:
            ValueError: If the Tron config has not been loaded yet (e.g., before `apply`).
        """
        if self._tron_config is None:
            raise ValueError("Tron config is not set")
        return self._tron_config

    @property
    def tokenizer(self) -> Optional["AutoTokenizer"]:
        """Get the Hugging Face tokenizer.

        Loads the tokenizer specified by `hf_tokenizer_path` during initialization
        or attempts to load one found within the NeMo checkpoint during `apply`.

        Returns:
            A Hugging Face AutoTokenizer instance, or None if not found/specified.
        """
        if self._tokenizer is None:
            if self._hf_tokenizer_path is not None:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self._hf_tokenizer_path, trust_remote_code=True)

        return self._tokenizer

    @abstractmethod
    def convert_state(self, source: Any, target: Any) -> None:
        """Convert the state dict from the source NeMo model to the target HF model.

        Must be implemented by subclasses.

        Args:
            source: A helper object (_ModelState) containing the loaded NeMo state dict.
            target: The target Hugging Face model instance.
        """
        raise NotImplementedError

    def init_hf_model(self, dtype: torch.dtype = torch.bfloat16) -> "AutoModelForCausalLM":
        """Initialize the target Hugging Face model without initializing weights.

        Args:
            dtype (torch.dtype, optional): The desired data type for the model.
                                        Defaults to torch.bfloat16.

        Returns:
            An initialized Hugging Face AutoModelForCausalLM instance.
        """
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights(True):
            return AutoModelForCausalLM.from_config(self.hf_config, torch_dtype=dtype)

    def init_tron_model(self) -> tuple[dict[str, torch.Tensor], GPTConfig | T5Config]:
        """Load the NeMo Tron model state dict and config from a distributed checkpoint.

        Loads the full state dict directly without initializing the full NeMo model
        for memory efficiency.

        Returns:
            A tuple containing:
            - The loaded, unsharded NeMo state dict.
            - The loaded NeMo model configuration (e.g., GPTConfig).
        """
        tron_yaml = self.input_path / "run_config.yaml"
        assert tron_yaml.exists(), f"Tron config file {tron_yaml} does not exist"
        with open(tron_yaml, "r") as stream:
            _config = yaml.safe_load(stream)
        model_config = _config["model_config"]
        model_config = instantiate(model_config)

        if self._hf_tokenizer_path is None:
            # Try to build tokenizer from the NeMo checkpoint
            tokenizer_config: TokenizerConfig | None = instantiate(_config["tokenizer_config"])
            if (
                tokenizer_config is not None
                and tokenizer_config.tokenizer_type == "HuggingFaceTokenizer"
                and tokenizer_config.tokenizer_model is not None
                and Path(tokenizer_config.tokenizer_model).exists()
            ):
                self._hf_tokenizer_path = tokenizer_config.tokenizer_model
            else:
                logger.warning("Failed to find Huggingface tokenizer in the NeMo checkpoint")

        state_dict = {}
        state_dict = get_full_mcore_state_dict(self.input_path, model_cfg=model_config)

        return state_dict, model_config

    def apply(self) -> Path:
        """Run the full conversion process from NeMo Tron to Hugging Face.

        Loads the NeMo state dict and config, initializes the target HF model,
        converts the state dict, and saves the HF model and tokenizer.

        Returns:
            Path to the saved Hugging Face model directory.
        """
        logger.info("Loading Tron checkpoint. This may take a while...")
        state_dict, source_config = self.init_tron_model()
        self._tron_config = source_config
        logger.info("Tron checkpoint loaded.")

        self.config = self.hf_config  # for backward compatibility
        source = _ModelState(state_dict)
        source.config = self.tron_config

        target = self.init_hf_model(torch_dtype_from_mcore_config(source_config))
        target = self.convert_state(source, target)

        target = target.cpu()
        if self.hf_config.tie_word_embeddings:
            state_dict = target.state_dict()
            state_dict.pop("lm_head.weight")
            target.save_pretrained(self.output_path, state_dict=state_dict)
        else:
            target.save_pretrained(self.output_path)

        try:
            self.tokenizer.save_pretrained(self.output_path)
        except Exception:
            logger.warning("Failed to save tokenizer")

        if self.tron_config.generation_config is not None:
            self.tron_config.generation_config.save_pretrained(self.output_path)

        print(f"Converted {self.input_path} to {self.output_path}.")
        return self.output_path
