"""
Test the LLaMA3 recipe with a smaller model.
"""

import argparse

import nemo_run as run
import torch

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers import SentencePieceTokenizer


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
    parser.add_argument('--sp', type=bool, default=None, help="Override sequence parallel")

    return parser.parse_args()


def train_data(data_path, tokenizer_path, index_mapping_dir, seq_length):
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


def small_model_cfg(seq_length):
    """Small 145m model"""
    return llm.Llama3Config(
        rotary_base=500_000,
        seq_length=seq_length,
        num_layers=12,
        hidden_size=768,
        ffn_hidden_size=2688,
        num_attention_heads=16,
        init_method_std=0.023,
    )


def _create_verify_precision(precision: torch.dtype):
    def verify_precision(tensor: torch.Tensor) -> None:
        assert tensor.dtype == precision

    return verify_precision


def main():
    args = get_args()

    pretrain_recipe = llm.llama3_8b.pretrain_recipe(
        dir=args.exp_dir, name="L2_llama3_small_pretrain_test", num_gpus_per_node=args.devices
    )

    pretrain_recipe.model = llm.LlamaModel(small_model_cfg(args.seq_length))

    if args.data_path and args.tokenizer_path:
        pretrain_recipe.data = train_data(
            data_path=args.data_path,
            tokenizer_path=args.tokenizer_path,
            index_mapping_dir=args.index_mapping_dir,
            seq_length=args.seq_length,
        )

    debugger_callback = ParameterDebugger(
        param_fn=_create_verify_precision(torch.bfloat16),
        grad_fn=_create_verify_precision(torch.float32),
        log_on_hooks=["on_train_start", "on_train_end"],
    )
    pretrain_recipe.callbacks.append(debugger_callback)

    pretrain_recipe.trainer.max_steps = args.max_steps
    pretrain_recipe.trainer.log_every_n_steps = 1
    pretrain_recipe.log.ckpt.every_n_train_steps = 1
    pretrain_recipe.trainer.val_check_interval = 0.5

    parallelisms = {
        "tensor_model_parallel_size": args.tp,
        "pipeline_model_parallel_size": args.pp,
        "virtual_pipeline_model_parallel_size": args.vp,
        "context_parallel_size": args.cp,
        "sequence_parallel": args.sp,
    }
    for k, v in parallelisms.items():
        if v is not None:  # use recipe default if not specified
            setattr(pretrain_recipe.trainer.strategy, k, v)

    run.run(pretrain_recipe, direct=True)


if __name__ == '__main__':
    main()
