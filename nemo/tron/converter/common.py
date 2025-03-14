import os
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.dist_checkpointing.strategies.torch import TorchDistLoadShardedStrategy
from omegaconf import OmegaConf

from nemo.lightning import _strategy_lib
from nemo.tron.container.utils.instantiate import instantiate


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
