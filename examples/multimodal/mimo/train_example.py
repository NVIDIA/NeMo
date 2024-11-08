import argparse
import os
import sys

import torch
import wandb
from megatron.core.optimizer import OptimizerConfig
from transformers import AutoProcessor

from nemo import lightning as nl
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm import import_ckpt
from nemo.collections.multimodal.mimo.data.mock import MockDataModule
from nemo.collections.multimodal.mimo.model.base import BaseMimoConfig, BaseMimoModel, CustomMimoConfig
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
from nemo.utils.exp_manager import TimingCallback


def main(args):

    # wandb.init(project=args.wandb_project, name=args.name)
    # Global and micro batch sizes
    gbs = 4
    mbs = 1
    seq_length = 256

    tokenizer = AutoTokenizer("llava-hf/llava-v1.6-vicuna-7b-hf")
    special_tokens = [f"IMG_{i}" for i in range(8)]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    image_special_tokens = [f"IMG_{i}" for i in range(8)]
    image_special_token_indices = [tokenizer.tokenizer.convert_tokens_to_ids(f"IMG_{i}") for i in range(8)]

    data = MockDataModule(
        tokenizer=tokenizer, vocab_size=tokenizer.vocab_size, micro_batch_size=mbs, global_batch_size=gbs
    )

    # Training strategy setup
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        pipeline_dtype=torch.bfloat16,
    )

    # Checkpoint callback setup
    checkpoint_callback = nl.ModelCheckpoint(
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=2,
        every_n_train_steps=200,
        dirpath=args.log_dir,
    )

    trainer = nl.Trainer(
        devices=args.devices,
        max_steps=10000,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="32"),
        callbacks=[checkpoint_callback, TimingCallback()],
        val_check_interval=50,
        limit_val_batches=gbs,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    custom_config = CustomMimoConfig(
        vocab_size=tokenizer.vocab_size,
        image_special_token_indices=image_special_token_indices,
        image_special_tokens=image_special_tokens,
        freeze_language_model=False,
    )
    # base_config = BaseMimoConfig(vocab_size = tokenizer.vocab_size)
    model = BaseMimoModel(config=custom_config, tokenizer=tokenizer)

    # Logger setup
    from pytorch_lightning.loggers import WandbLogger

    nemo_logger = nl.NeMoLogger(
        log_dir=args.log_dir,
        name=args.name,
        wandb=WandbLogger(project=args.wandb_project, name=args.name) if args.wandb_project is not None else None,
    )
    nemo_logger.setup(
        trainer,
        resume_if_exists=True,
    )

    # Auto resume setup
    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        resume_from_directory=args.log_dir,
        # restore_config=None,
        restore_config=RestoreConfig(path=args.restore_path) if args.restore_path else None,
    )
    resume.setup(trainer, model)

    # Optimizer and scheduler setup
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=2.0e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        use_distributed_optimizer=False,
        bf16=True,
    )
    sched = CosineAnnealingScheduler(
        max_steps=trainer.max_steps,
        warmup_steps=70,
        constant_steps=0,
        min_lr=2.0e-4,
    )
    opt = MegatronOptimizerModule(opt_config, sched)
    opt.connect(model)

    trainer.fit(model, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mimo Model Training Script")

    # Argument parsing
    parser.add_argument(
        "--log_dir", type=str, required=False, default="./", help="Directory for logging and checkpoints"
    )
    parser.add_argument("--devices", type=int, required=False, default=8)
    parser.add_argument("--tp_size", type=int, required=False, default=2)
    parser.add_argument("--pp_size", type=int, required=False, default=1)
    parser.add_argument("--name", type=str, required=False, default="mimo_decoder_align")
    parser.add_argument("--wandb_project", type=str, required=False, default=None)
    parser.add_argument("--restore_path", type=str, required=False, default=None)

    args = parser.parse_args()
    main(args)

#  s s s s s s || x x x x s s
#  0 0 0 0 0 0 || x x x x 0 0

#  s s s s x x || x x s s s s
#  0 0 0 0 x x || x x 0 0 0 0
