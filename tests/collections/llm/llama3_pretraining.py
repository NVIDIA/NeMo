"""
Test the LLaMA3 recipe with a smaller model.
"""

import argparse

import nemo_run as run
import pytorch_lightning as pl
import torch

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers import SentencePieceTokenizer
from nemo.lightning.pytorch.callbacks.debugging import ParameterDebugger


def get_args():
    parser = argparse.ArgumentParser(prog="", description="")
    parser.add_argument('--devices', type=int, required=True, help="Number of devices to use for training")
    parser.add_argument('--max-steps', type=int, required=True, help="Number of steps to train for")
    parser.add_argument(
        '--experiment-dir', type=str, required=True, help="directory to write results and checkpoints to"
    )
    parser.add_argument(
        '--data-path', type=str, default=None, help="Path to data file. If not specified, uses mock data."
    )
    parser.add_argument(
        '--tokenizer-path',
        type=str,
        default=None,
        help="Path to a sentencepiece tokenizer model file. If not specified, uses mock data.",
    )
    parser.add_argument('--index-mapping-dir', type=str, help="directory to write index mappings to")
    parser.add_argument('--seq-length', type=int, default=8192, help="Sequence length. default is 8k")
    parser.add_argument('--tp', type=int, default=None, help="Override tensor parallelism")
    parser.add_argument('--pp', type=int, default=None, help="Override pipeline parallelism")
    parser.add_argument('--vp', type=int, default=None, help="Override virtual pipeline parallelism")
    parser.add_argument('--cp', type=int, default=None, help="Override context parallelism")
    parser.add_argument('--sp', type=int, choices=[0, 1], default=None, help="Override sequence parallel")
    parser.add_argument(
        '--precision', type=str, choices=['bf16', 'fp16', 'fp32'], default='bf16', help="Override recipe precision"
    )
    parser.add_argument('--fp8', action='store_true', help="Enable FP8")

    return parser.parse_args()


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


def small_model_cfg(seq_length: int) -> llm.GPTConfig:
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


def verify_distcp_dir(ckpt_path: str) -> None:
    pass


def verify_ckpt_dir(
    model_ckpt: nl.ModelCheckpoint, max_steps: int, val_check_interval: int, exp_dir: str, dist_ckpts: bool = True
) -> None:
    """Ensures a checkpoint directory has the correct number of checkpoints, followed top-k, a checkpoint
    for the last step exists, and the checkpoints are the correct format.
    """

    import os

    ckpts = os.listdir(os.path.join(exp_dir, 'checkpoints'))

    expected_ckpts = (max_steps // val_check_interval) + model_ckpt.save_last
    if model_ckpt.save_last:
        assert any([c.endswith('-last') for c in ckpts]), "No -last checkpoint found after training"
    if model_ckpt.save_top_k > 0:
        assert (
            len(ckpts) == expected_ckpts or len(ckpts) == model_ckpt.save_top_k + model_ckpt.save_last
        ), f"Expected {expected_ckpts} checkpoints, or at most top {model_ckpt.save_top_k}"
    else:
        assert len(ckpts) == expected_ckpts, f"Expected {expected_ckpts} checkpoints"

    for ckpt_path in ckpts:
        assert os.path.isdir(ckpt_path) == dist_ckpts, "Checkpoint is not correct type"

        if ckpt_path.endswith('-last') and 'step' in model_ckpt.filename:
            assert f'step={max_steps}' in ckpt_path

        if dist_ckpts:
            verify_distcp_dir(ckpt_path)


def _create_verify_precision(precision: torch.dtype):
    def verify_precision(tensor: torch.Tensor) -> None:
        assert tensor.dtype == precision

    return verify_precision


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


def main():
    args = get_args()

    pretrain_recipe = llm.llama3_8b.pretrain_recipe(
        dir=args.experiment_dir, name="L2_llama3_small_pretrain_test", num_gpus_per_node=args.devices
    )

    pretrain_recipe.model = llm.LlamaModel(small_model_cfg(args.seq_length))

    if args.data_path and args.tokenizer_path:
        pretrain_recipe.data = train_data(
            data_path=args.data_path,
            tokenizer_path=args.tokenizer_path,
            index_mapping_dir=args.index_mapping_dir,
            seq_length=args.seq_length,
        )

    # Recipe Overrides
    pretrain_recipe.trainer.max_steps = args.max_steps
    pretrain_recipe.trainer.log_every_n_steps = 1
    pretrain_recipe.log.ckpt.every_n_train_steps = None
    pretrain_recipe.trainer.val_check_interval = 2

    if not args.precision == 'bf16' or args.fp8:  # default case is bf16 without fp8
        import llm.recipes.precision.mixed_precision as mp_recipes

        key = (args.precision, args.fp8)
        precision_recipe = {
            ("fp16", False): mp_recipes.fp16_mixed,
            ("bf16", True): mp_recipes.bf16_with_fp8_mixed,
            ("fp16", True): mp_recipes.fp16_with_fp8_mixed,
            # Need fp32
        }[key]
        pretrain_recipe.plugins = precision_recipe()
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    debugger_callback = ParameterDebugger(
        param_fn=_create_verify_precision(dtype_map[args.precision]),
        grad_fn=_create_verify_precision(torch.float32),
        log_on_hooks=["on_train_start", "on_train_end"],
    )
    pretrain_recipe.trainer.callbacks.append(debugger_callback)

    parallelisms = {
        "tensor_model_parallel_size": args.tp,
        "pipeline_model_parallel_size": args.pp,
        "virtual_pipeline_model_parallel_size": args.vp,
        "context_parallel_size": args.cp,
        "sequence_parallel": bool(args.sp) if args.sp is not None else None,
    }
    for k, v in parallelisms.items():
        if v is not None:  # use recipe default if not specified
            setattr(pretrain_recipe.trainer.strategy, k, v)
        parallelisms[k] = getattr(pretrain_recipe.trainer.strategy, k)
    pretrain_recipe.trainer.callbacks.append(ModelAttrValidationCallback(parallelisms))

    run.run(pretrain_recipe, direct=True)

    verify_ckpt_dir(
        pretrain_recipe.log.ckpt, args.max_steps, pretrain_recipe.trainer.val_check_interval, args.experiment_dir
    )


if __name__ == '__main__':
    main()
