import argparse
import os
import sys

import requests
import torch
from megatron.core.optimizer import OptimizerConfig
from megatron.energon import VQASample
from PIL import Image
from transformers import AutoProcessor

from nemo import lightning as nl
from nemo.collections import llm, vlm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.multimodal.data.energon import SimpleMultiModalDataModule
from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
from nemo.collections.vlm import ImageDataConfig, Llava1_5Config7B, LlavaModel, LlavaNextTaskEncoder
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.utils import logging
from nemo.utils.exp_manager import TimingCallback


def main(args):
    # Global and micro batch sizes
    gbs = 32
    mbs = 4
    seq_length = 256

    processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
    data_path = args.data_path
    # data_path = '/lustre/fsw/coreai_dlalgo_genai/datasets/energon_datasets/LLaVA-Pretrain-LCS-558K'
    image_processor = processor.image_processor
    # tokenizer = processor.tokenizer

    tokenizer = AutoTokenizer("llava-hf/llava-v1.6-vicuna-7b-hf")
    # tokenizer = processor.tokenizer

    tokenizer = AutoTokenizer("llava-hf/llava-v1.6-vicuna-7b-hf")

    multimodal_sample_config = MultiModalSampleConfig()

    task_encoder = LlavaNextTaskEncoder(
        tokenizer=tokenizer.tokenizer,
        image_processor=image_processor,
        multimodal_sample_config=multimodal_sample_config,
    )
    data = SimpleMultiModalDataModule(
        path=data_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
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
        every_n_train_steps=500,
        dirpath=args.log_dir,
    )

    trainer = nl.Trainer(
        devices=args.devices,
        max_steps=2170,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        callbacks=[checkpoint_callback, TimingCallback()],
        val_check_interval=100,
        limit_val_batches=gbs,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    from nemo.collections.llm import import_ckpt
    from nemo.collections.llm.gpt.model.llama import LlamaModel

    language_transformer_config = llm.Llama2Config7B()

    # import_ckpt(
    #     model=LlamaModel(language_transformer_config),
    #     source='hf://lmsys/vicuna-7b-v1.5',
    # )

    vision_transformer_config = vlm.HFCLIPVisionConfig(
        pretrained_model_name_or_path="openai/clip-vit-large-patch14-336"
    )
    vision_projection_config = vlm.MultimodalProjectorConfig(
        projector_type=args.projector_type,
        input_size=1024,
        hidden_size=4096,
        ffn_hidden_size=4096,
    )

    # NEVA model configuration
    neva_config = vlm.NevaConfig(
        language_transformer_config=language_transformer_config,
        vision_transformer_config=vision_transformer_config,
        vision_projection_config=vision_projection_config,
        language_model_from_pretrained=args.language_model_path,
        freeze_language_model=True,
    )

    model = vlm.NevaModel(neva_config, tokenizer=data.tokenizer)

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
    parser = argparse.ArgumentParser(description="NEVA Model Training Script")

    # Argument parsing
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset JSON file")
    parser.add_argument("--log_dir", type=str, required=True, help="Directory for logging and checkpoints")
    parser.add_argument("--devices", type=int, required=False, default=1)
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--pp_size", type=int, required=False, default=1)
    parser.add_argument(
        "--language_model_path", type=str, required=False, default=None, help="Path to the pretrained language model"
    )
    parser.add_argument("--name", type=str, required=False, default="llava_next_test")
    parser.add_argument("--wandb_project", type=str, required=False, default=None)
    parser.add_argument("--projector_type", type=str, required=False, default="mlp2x_gelu")

    args = parser.parse_args()
    main(args)
