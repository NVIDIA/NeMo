import pytorch_lightning as pl
import torch

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers import SentencePieceTokenizer


def train_data(
    data_path: str, tokenizer_path: str, index_mapping_dir: str, seq_length: int
) -> llm.PreTrainingDataModule:
    """Single shard dataset tokenized by SentencePiece"""
    tokenizer = SentencePieceTokenizer(model_path=tokenizer_path)
    return llm.PreTrainingDataModule(
        paths=data_path,
        tokenizer=tokenizer,
        seq_length=seq_length,
        micro_batch_size=4,
        global_batch_size=32,
        seed=1234,
        index_mapping_dir=index_mapping_dir,
    )


def small_llama_cfg(seq_length: int) -> llm.GPTConfig:
    """Small 145m model"""
    return llm.Llama3Config8B(
        rotary_base=500_000,
        seq_length=seq_length,
        num_layers=12,
        hidden_size=768,
        ffn_hidden_size=2688,
        num_attention_heads=16,
        init_method_std=0.023,
    )


class MCoreModelAttributeValidator(pl.Callback):
    """Walk through submodules and verify user-specified attributes like parallelisms."""

    def __init__(self, attr_dict: dict):
        super().__init__()
        self.attr_dict = attr_dict

    def _check_attrs(self, target):
        for k, v in self.attr_dict.items():
            if hasattr(target, k):
                model_val = getattr(target, k)
                assert (
                    model_val == v
                ), f"Key {k} for model ({model_val}) does not match {v} from provided attribute mapping."

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        def walk_fn(module: torch.nn.Module) -> torch.nn.Module:
            self._check_attrs(module)
            if hasattr(module, "config"):
                self._check_attrs(module.config)

            return module

        trainer.model.walk(walk_fn)


def verify_distcp_dir(ckpt_path: str) -> None:
    pass


def verify_ckpt_dir(
    model_ckpt: nl.ModelCheckpoint, max_steps: int, val_check_interval: int, exp_dir: str, dist_ckpts: bool = True
) -> None:
    """Ensures a checkpoint directory has the correct number of checkpoints, followed top-k, a checkpoint
    for the last step exists, and the checkpoints are the correct format.
    """

    import os

    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    ckpts = os.listdir(ckpt_dir)

    expected_ckpts = (max_steps // val_check_interval) + model_ckpt.save_last
    if model_ckpt.save_last:
        assert any([c.endswith('-last') for c in ckpts]), "No -last checkpoint found after training"
    if model_ckpt.save_top_k > 0:
        assert (
            len(ckpts) == expected_ckpts or len(ckpts) == model_ckpt.save_top_k + model_ckpt.save_last
        ), f"Expected {expected_ckpts} checkpoints, or at most top {model_ckpt.save_top_k}"
    else:
        assert len(ckpts) == expected_ckpts, f"Expected {expected_ckpts} checkpoints"

    for ckpt_name in ckpts:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if ckpt_name.endswith('-last') and 'step' in model_ckpt.filename:
            assert f'step={max_steps-1}' in ckpt_name

        if dist_ckpts:
            assert os.path.isdir(ckpt_path), "Checkpoint is not correct type"
            verify_distcp_dir(ckpt_name)
        else:
            assert os.path.isfile(ckpt_path), "Checkpoint is not correct type"


def create_verify_precision(precision: torch.dtype):
    def verify_precision(tensor: torch.Tensor) -> None:
        assert tensor.dtype == precision

    return verify_precision
