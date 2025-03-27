import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import yaml
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.strategies.torch import TorchDistLoadShardedStrategy
from megatron.core.optimizer import OptimizerConfig
from omegaconf import OmegaConf

from nemo.collections.llm.gpt.model.base import GPTConfig, torch_dtype_from_mcore_config
from nemo.collections.llm.t5.model.t5 import T5Config
from nemo.lightning import _strategy_lib
from nemo.tron.checkpointing import save_checkpoint
from nemo.tron.config import CheckpointConfig, ConfigContainer, LoggerConfig
from nemo.tron.state import GlobalState
from nemo.tron.tokenizers.tokenizer import build_tokenizer
from nemo.tron.utils.instantiate_utils import instantiate

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, PretrainedConfig


logger = logging.getLogger(__name__)


@contextmanager
def temporary_distributed_context():
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="gloo", world_size=1, rank=0)
    parallel_state.initialize_model_parallel()
    yield
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def get_full_mcore_state_dict(dist_ckpt_folder: Path):
    with temporary_distributed_context():
        cfg = OmegaConf.load(dist_ckpt_folder / "run_config.yaml")
        cfg = cfg.model_config
        model_cfg = instantiate(cfg)
        model_cfg.params_dtype = torch.bfloat16
        with _strategy_lib.megatron_cpu_init_context(model_cfg):
            model = model_cfg.configure_model(None)

        strategy = TorchDistLoadShardedStrategy()
        state_dict = strategy.load(model.sharded_state_dict(), Path(dist_ckpt_folder))
        del model

    return state_dict


def save_hf_tokenizer_assets(tokenizer_name_or_path, save_path="/tmp/nemo_tokenizer"):
    """Save HF tokenizer to the imported NeMo model"""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    # Save tokenizer assets to save_path.
    tok.save_pretrained(save_path)
    return save_path


def dtype_from_str(dtype):
    """
    Convert a str precision to equivalent torch dtype.
    """
    assert isinstance(dtype, str)
    if dtype in ["float16", "fp16", "16", "16-mixed"]:
        return torch.float16
    elif dtype in ["bfloat16", "bf16-mixed"]:
        return torch.bfloat16
    else:
        return torch.float32


def dtype_from_hf(config):
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


class BaseImporter:
    def __init__(self, input_path: str | Path, output_path: str | Path):
        self.input_path = Path(input_path) if isinstance(input_path, str) else input_path
        self.output_path = Path(output_path) if isinstance(output_path, str) else output_path

    def init_tron_model(self, cfg: GPTConfig | T5Config):
        with _strategy_lib.megatron_cpu_init_context(cfg):
            model = cfg.configure_model(tokenizer=self.tokenizer)
        return [model]

    def init_hf_model(self):
        raise NotImplementedError

    def convert_state(self, source, target):
        raise NotImplementedError

    @property
    def config(self) -> GPTConfig | T5Config:
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
            target = self.init_tron_model(self.config)
            self.convert_state(source, target[0])
            state = GlobalState()
            state.cfg = ConfigContainer(
                model_config=self.config,
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
                ft_config=None,
                straggler_config=None,
                profiling_config=None,
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


class BaseExporter:
    def __init__(self, input_path: str | Path, output_path: str | Path):
        self.input_path = Path(input_path) if isinstance(input_path, str) else input_path
        self.output_path = Path(output_path) if isinstance(output_path, str) else output_path

    @property
    def config(self) -> "PretrainedConfig":
        raise NotImplementedError

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

        with no_init_weights(True):
            return AutoModelForCausalLM.from_config(self.config, torch_dtype=dtype)

    def init_tron_model(self) -> tuple[dict, dict]:
        """
        This function loads the state dict directly from a distributed checkpoint, and modify the state dict
        so that it is consistent with the key names you would get from loading the checkpoint into a model.
        This is a more memory-efficient method to obtain a state dict without initializing the nemo model.

        Args:
            path (Path): The path from which the model will be loaded.

        Returns
        -------
            tuple[Dict, Dict]: The loaded state dict and the yaml config dict.
        """
        tron_yaml = self.input_path / "run_config.yaml"
        assert tron_yaml.exists()
        with open(tron_yaml, "r") as stream:
            _config = yaml.safe_load(stream)
        config = _config["model_config"]
        config = instantiate(config)
        try:
            self.tokenizer = build_tokenizer(instantiate(_config["tokenizer_config"]))
        except Exception:
            logger.warning("Failed to build tokenizer")

        state_dict = {}
        state_dict = get_full_mcore_state_dict(self.input_path)

        return state_dict, config

    def apply(self) -> Path:
        """Run the conversion from Tron to HF format."""
        logger.info("Loading Tron checkpoint. This may take a while...")
        state_dict, source_config = self.init_tron_model()
        self._source_config = source_config
        logger.info("Tron checkpoint loaded.")

        source = _ModelState(state_dict)
        target = self.init_hf_model(torch_dtype_from_mcore_config(source_config))
        target = self.convert_state(source, target)

        target = target.cpu()
        if self.config.tie_word_embeddings:
            state_dict = target.state_dict()
            state_dict.pop("lm_head.weight")
            target.save_pretrained(self.output_path, state_dict=state_dict)
        else:
            target.save_pretrained(self.output_path)

        try:
            self.tokenizer._tokenizer.save_pretrained(self.output_path)
        except Exception:
            logger.warning("Failed to save tokenizer")

        return self.output_path
