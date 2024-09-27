import argparse
import os
import torch

from pathlib import Path

from nemo.lightning import Trainer, MegatronStrategy
from nemo.collections.llm import PreTrainingDataModule, MixtralConfig8x3B, MixtralModel

from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule as MegatronOptim, OptimizerConfig
from megatron.core.distributed import DistributedDataParallelConfig as McoreDDPConfig
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.llm.api import train
from nemo.lightning import NeMoLogger



def tokenizer(vocab_path, merges_path):
    return get_nmt_tokenizer(
        "megatron",
        "GPT2BPETokenizer",
        vocab_file=vocab_path,
        merges_file=merges_path,
    )

def main(args):
    strategy = MegatronStrategy(
        expert_model_parallel_size=args.devices,
        tensor_model_parallel_size=1,
        sequence_parallel=True,
        context_parallel_size=1,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_dtype=torch.float32,
        precision=torch.bfloat16,
        ddp=McoreDDPConfig(
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=False,
                use_distributed_optimizer=True,
                check_for_nan_in_grad=True,
                bucket_size=None,
            )
    )

    trainer = Trainer(
        log_every_n_steps=1,
        devices=args.devices,
        max_steps=4,
        accelerator="gpu",
        strategy=strategy,
        num_sanity_val_steps=0,
        logger=None,
        limit_val_batches=1,
    )

    data = PreTrainingDataModule(
        args.data_path,
        seq_length=512,
        global_batch_size=2,
        micro_batch_size=1,
        num_workers=1,
        split='99,1,0',
        tokenizer=tokenizer(args.vocab_path, args.merges_path),
    )

    mixtral_config = MixtralConfig8x3B(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=8,
        num_query_groups=8,
        ffn_hidden_size=320,
        kv_channels=16,
        init_method_std=0.015,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        layernorm_epsilon=1e-5,
        make_vocab_size_divisible_by=128,
        max_position_embeddings=512,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
    )
    mixtral_config.overlap_param_gather_with_optimizer_step=True

    optim_config = OptimizerConfig(
        fp16=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        lr=0.01,
        weight_decay=0,
        adam_beta1=0.9,
        adam_beta2=0.9,
        clip_grad=0.0,
        use_distributed_optimizer=True,
        min_lr=0.0,
        log_num_zeros_in_grad=True,
        barrier_with_L1_time=True,
    )

    opt = MegatronOptim(config=optim_config)
    model = MixtralModel(mixtral_config, optim=opt, tokenizer=data.tokenizer)

    nemo_logger = NeMoLogger(
        name=args.experiment_name,
        use_datetime_version=False,
        explicit_log_dir=args.experiment_dir,
    )

    output_path = Path(args.experiment_dir)
    assert not output_path.exists(), f"Did not expect {output_path} to exist"

    train(
        model=model,
        resume=None,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        tokenizer='data',
        optim=opt,
    )

    output_path = Path(args.experiment_dir) / "checkpoints/--None=0.0000-epoch=0/"
    assert output_path.exists(), f"Expected {output_path} to exist"
    assert output_path.is_dir(), f"Expected {output_path} to be a directory"
    output_files = ['__0_0.distcp', '__0_1.distcp', 'common.pt', 'metadata.json', '.metadata']
    for file in output_files:
        path = output_path / file
        assert path.exists(), f"Expected {file} to exist"
        assert path.is_file(), f"Expected {file} to be a file"
        assert os.access(path, os.R_OK), f"Expected {file} to be readable"
        assert path.stat().st_size, f"Expected {file} to be non-empty"

    for file in os.listdir(output_path):
        assert file in output_files, f"Got unexpected {file} in checkpoint directory"

def parse_args():
    parser = argparse.ArgumentParser(description='Train a small Mixtral model using NeMo 2.0')
    parser.add_argument('--devices', type=int, default=1, help="Number of devices to use for training")
    parser.add_argument('--max-steps', type=int, default=4, help="Number of steps to train for")
    parser.add_argument('--experiment-dir', type=str, default='/tmp/exp_dir', help="directory to write results and checkpoints to")
    parser.add_argument('--experiment-name', type=str, default='mini_mixtral_test', help="name of experiment")
    parser.add_argument('--data-path', type=str, help="Path to data file")
    parser.add_argument('--vocab-path', type=str, default=None, help="Path to vocab file")
    parser.add_argument('--merges-path', type=str, default=None, help="Path to merges file")

    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
