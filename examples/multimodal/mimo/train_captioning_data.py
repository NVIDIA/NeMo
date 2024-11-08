import argparse
import faulthandler
import logging
import os
import sys

import torch
from megatron.core.optimizer import OptimizerConfig
from transformers import AutoProcessor

from nemo import lightning as nl
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm import import_ckpt
from nemo.collections.multimodal.data.energon import SimpleMultiModalDataModule
from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
from nemo.collections.multimodal.mimo.data.captioning import MimoCaptioningTaskEncoder
from nemo.collections.multimodal.mimo.model.base import BaseMimoConfig, BaseMimoModel, CustomMimoConfig
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback

logging.basicConfig(level=logging.DEBUG)
faulthandler.enable()
# Optionally, redirect output to a specific file
# faulthandler.enable(file=open("faulthandler.log", "w"))


def main(args):
    # Global and micro batch sizes
    gbs = 128
    mbs = 32
    seq_length = 256
    data_path = '/lustre/fsw/coreai_dlalgo_genai/ykarnati/datasets/cc3m-wds'
    tokenizer = AutoTokenizer("llava-hf/llava-v1.6-vicuna-7b-hf")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
    special_tokens = [f"IMG_{i}" for i in range(8)]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    image_special_tokens = [f"IMG_{i}" for i in range(8)]
    image_special_token_indices = [tokenizer.tokenizer.convert_tokens_to_ids(f"IMG_{i}") for i in range(8)]

    multimodal_sample_config = MultiModalSampleConfig()

    task_encoder = MimoCaptioningTaskEncoder(
        tokenizer=tokenizer.tokenizer,
        image_processor=processor.image_processor,
        multimodal_sample_config=multimodal_sample_config,
    )
    data = SimpleMultiModalDataModule(
        path=data_path,
        tokenizer=tokenizer,
        image_processor=processor.image_processor,
        num_workers=8,
        micro_batch_size=mbs,
        global_batch_size=gbs,
        multimodal_sample_config=multimodal_sample_config,
        task_encoder=task_encoder,
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
        every_n_train_steps=300,
        dirpath=args.log_dir,
    )

    trainer = nl.Trainer(
        devices=args.devices,
        max_steps=3500,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="32"),
        callbacks=[checkpoint_callback, TimingCallback()],
        val_check_interval=30,
        limit_val_batches=gbs,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    custom_config = CustomMimoConfig(
        vocab_size=tokenizer.vocab_size,
        image_special_token_indices=image_special_token_indices,
        image_special_tokens=image_special_tokens,
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
        restore_config=None,
    )
    resume.setup(trainer, model)

    # Optimizer and scheduler setup
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=0.001,
        adam_beta1=0.9,
        adam_beta2=0.95,
        use_distributed_optimizer=False,
        bf16=True,
    )
    sched = CosineAnnealingScheduler(
        max_steps=trainer.max_steps,
        warmup_steps=70,
        constant_steps=0,
        min_lr=2.0e-05,
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
    parser.add_argument("--name", type=str, required=False, default="mimo_first_light")
    parser.add_argument("--wandb_project", type=str, required=False, default=None)

    args = parser.parse_args()
    main(args)

#  s s s s s s || x x x x s s
#  0 0 0 0 0 0 || x x x x 0 0

#  s s s s x x || x x s s s s
#  0 0 0 0 x x || x x 0 0 0 0
