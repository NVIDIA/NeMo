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
from megatron.core.dist_checkpointing.strategies.torch import \
    TorchDistLoadShardedStrategy
from megatron.core.optimizer import OptimizerConfig

from nemo.collections.llm.gpt.model.base import (GPTConfig,
                                                 torch_dtype_from_mcore_config)
from nemo.collections.llm.t5.model.t5 import T5Config
from nemo.tron.checkpointing import save_checkpoint
from nemo.tron.config import (CheckpointConfig, ConfigContainer, LoggerConfig,
                              TokenizerConfig)
from nemo.tron.state import GlobalState
from nemo.tron.tokenizers.tokenizer import _HuggingFaceTokenizer
from nemo.tron.utils.instantiate_utils import instantiate

if TYPE_CHECKING:
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              PretrainedConfig)


logger = logging.getLogger(__name__)

HF_ASSETS_DIR = "hf_assets"


def materialize_meta_tensors(obj: Any) -> Any:  # noqa: ANN401
    """Recursively replace *meta* tensors with real CPU tensors.

    This is handy when models are initialized on the *meta* device.
    Such tensors lack storage and need to be converted before any real computation or serialization.

    The function walks through common container types (``dict``, ``list``, ``tuple``)
    and converts any tensor residing on the *meta* device to a freshly allocated
    ``torch.empty`` tensor on the CPU that keeps the same ``shape`` and ``dtype``.

    Args:
        obj: Arbitrary python object that may contain ``torch.Tensor`` instances.

    Returns
    -------
        The same object with all *meta* tensors materialised on CPU.  For immutable
        containers like tuples a new instance is returned; mutable containers are
        modified in-place for efficiency.
    """

    # Fast path: raw tensor
    if isinstance(obj, torch.Tensor):
        if obj.device.type == "meta":
            return torch.empty(obj.size(), dtype=obj.dtype, device="cpu")
        return obj

    # Handle wrappers/Parameters holding a tensor in `.data`
    if hasattr(obj, "data") and isinstance(obj.data, torch.Tensor):
        if obj.data.device.type == "meta":
            obj.data = torch.empty(obj.data.size(), dtype=obj.data.dtype, device="cpu")
        return obj

    # Mapping (dict-like)
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = materialize_meta_tensors(v)
        return obj

    # Sequence types
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = materialize_meta_tensors(obj[i])
        return obj

    if isinstance(obj, tuple):
        return tuple(materialize_meta_tensors(v) for v in obj)

    # Anything else is returned as-is
    return obj


@contextmanager
def megatron_cpu_init_context(config) -> Generator[None, None, None]:
    """ """
    _orig_use_cpu_initialization = config.use_cpu_initialization

    config.use_cpu_initialization = True

    yield

    config.use_cpu_initialization = _orig_use_cpu_initialization


@contextmanager
def megatron_lazy_init_context(config) -> Generator[None, None, None]:
    """ """
    try:
        from megatron.core.extensions import transformer_engine as _te

        original = _te._get_extra_te_kwargs  # noqa: SLF001

        def _get_extra_te_kwargs_meta(c):
            """Forces device to meta"""
            kwargs = original(c)
            kwargs["device"] = "meta"
            return kwargs

        _te._get_extra_te_kwargs = _get_extra_te_kwargs_meta  # noqa: SLF001
    except ImportError:
        pass

    _orig_perform_initialization = config.perform_initialization
    _orig_use_cpu_initialization = config.use_cpu_initialization

    config.perform_initialization = False
    config.use_cpu_initialization = True

    yield

    try:
        from megatron.core.extensions import transformer_engine as _te

        _te._get_extra_te_kwargs = original  # noqa: SLF001
    except ImportError:
        pass

    config.perform_initialization = _orig_perform_initialization
    config.use_cpu_initialization = _orig_use_cpu_initialization


@contextmanager
def temporary_distributed_context():
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


def get_full_mcore_state_dict(dist_ckpt_folder: Path, model_cfg):
    with temporary_distributed_context():
        if model_cfg.params_dtype != torch_dtype_from_mcore_config(model_cfg):
            logger.info(
                f"Converting params_dtype from {model_cfg.params_dtype} to {torch_dtype_from_mcore_config(model_cfg)}"
            )
            model_cfg.params_dtype = torch_dtype_from_mcore_config(model_cfg)

        with megatron_lazy_init_context(model_cfg):
            model = model_cfg.configure_model(None)

        strategy = TorchDistLoadShardedStrategy()
        sharded_state_dict = model.sharded_state_dict()
        sharded_state_dict = materialize_meta_tensors(sharded_state_dict)
        state_dict = strategy.load(sharded_state_dict, Path(dist_ckpt_folder))
        del model

    return state_dict


def save_hf_tokenizer_assets(tokenizer_name_or_path, save_path="/tmp/nemo_tokenizer"):
    """Save HF tokenizer to the imported NeMo model"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    # Save tokenizer assets to save_path.
    tok.save_pretrained(save_path)
    return save_path


def dtype_from_str(dtype: str) -> torch.dtype:
    """
    Convert a str precision to equivalent torch dtype.
    """
    assert isinstance(dtype, str)
    if dtype in ("float16", "fp16", "16", "16-mixed"):
        return torch.float16
    elif dtype in ("bfloat16", "bf16-mixed"):
        return torch.bfloat16
    else:
        return torch.float32


def dtype_from_hf(config) -> torch.dtype:
    """
    Extracts torch dtype from a HF config
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
    """
    Helper class for used for to modify state dict of a source model during model conversion.
    """

    def __init__(self, state_dict):
        self._state_dict = state_dict

    def state_dict(self):
        # pylint: disable=C0115,C0116
        return self._state_dict

    def to(self, dtype):
        # pylint: disable=C0115,C0116
        for k, v in self._state_dict.items():
            if v.dtype != dtype:
                logger.warning(f"Converting {k} from {v.dtype} (source model) to {dtype} (target model)")
            self._state_dict[k] = v.to(dtype)


class BaseImporter(ABC):
    def __init__(self, input_path: str | Path, output_path: str | Path):
        self.input_path = Path(input_path) if isinstance(input_path, str) else input_path
        self.output_path = Path(output_path) if isinstance(output_path, str) else output_path
        (self.output_path / HF_ASSETS_DIR).mkdir(parents=True, exist_ok=True)
        self._hf_config = None
        self._tron_config = None
        self.hf_config.save_pretrained(str(self.output_path / HF_ASSETS_DIR))

    @property
    def tokenizer(self) -> "_HuggingFaceTokenizer":
        """Get the tokenizer for the HF model.

        Returns:
            _HuggingFaceTokenizer: Tokenizer instance initialized from the HF model's tokenizer
        """

        return _HuggingFaceTokenizer(
            save_hf_tokenizer_assets(str(self.input_path), str(self.output_path / HF_ASSETS_DIR))
        )

    def init_tron_model(self, cfg: GPTConfig | T5Config):
        with megatron_lazy_init_context(cfg):
            model = cfg.configure_model(tokenizer=self.tokenizer)
        return [model]

    @abstractmethod
    def init_hf_model(self):
        raise NotImplementedError

    @abstractmethod
    def convert_state(self, source, target):
        raise NotImplementedError

    @property
    @abstractmethod
    def hf_config(self) -> "PretrainedConfig":
        raise NotImplementedError

    @property
    @abstractmethod
    def tron_config(self) -> GPTConfig | T5Config:
        raise NotImplementedError

    def apply(self) -> Path:
        """Run the conversion from HF to NeMo format.

        Args:
            output_path: Path where the converted model will be saved

        Returns:
            Path: Path to the saved NeMo model
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
    def __init__(self, input_path: str | Path, output_path: str | Path, hf_tokenizer_path: Optional[str] = None):
        self.input_path = Path(input_path) if isinstance(input_path, str) else input_path
        self.output_path = Path(output_path) if isinstance(output_path, str) else output_path
        self._hf_config = None
        self._tron_config = None
        self._hf_tokenizer_path = hf_tokenizer_path
        self._tokenizer = None

    @property
    @abstractmethod
    def hf_config(self) -> "PretrainedConfig":
        raise NotImplementedError

    @property
    def tron_config(self) -> GPTConfig | T5Config:
        if self._tron_config is None:
            raise ValueError("Tron config is not set")
        return self._tron_config

    @property
    def tokenizer(self) -> "AutoTokenizer":
        if self._tokenizer is None:
            if self._hf_tokenizer_path is not None:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self._hf_tokenizer_path, trust_remote_code=True)

        return self._tokenizer

    @abstractmethod
    def convert_state(self, source, target):
        raise NotImplementedError

    def init_hf_model(self, dtype=torch.bfloat16) -> "AutoModelForCausalLM":
        """Initialize a new Hugging Face Llama model with the specified data type.

        Args:
            dtype (torch.dtype): The data type for the model parameters. Default: torch.bfloat16

        Returns:
            LlamaForCausalLM: An initialized Hugging Face Llama model with no weights loaded
        """
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return AutoModelForCausalLM.from_config(self.hf_config, torch_dtype=dtype)

    def init_tron_model(self) -> tuple[dict, dict]:
        """
        This function loads the state dict directly from a distributed checkpoint, and modifies the state dict
        so that it is consistent with the key names you would get from loading the checkpoint into a model.
        This is a more memory-efficient method to obtain a state dict without initializing the nemo model.

        Args:
            path (Path): The path from which the model will be loaded.

        Returns
        -------
            tuple[Dict, Dict]: The loaded state dict and the yaml config dict.
        """
        tron_yaml = self.input_path / "run_config.yaml"
        assert tron_yaml.exists(), f"Tron config file {tron_yaml} does not exist"
        with open(tron_yaml, "r") as stream:
            _config = yaml.safe_load(stream)
        model_config = _config["model_config"]

        # NOTE: when loading the config from a trained checkpoint, the parallelism gets inferred from the training config.
        # This causes problems when we want to convert the model on a smaller number of GPUs than was used for training.
        # A WAR for models trained using PP > 1 is to set the pipeline parallelism to 1 here.
        model_config["pipeline_model_parallel_size"] = 1

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
        """Run the conversion from Tron to HF format."""
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
