import argparse

import torch
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoProcessor

import wandb
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm import Llama2Config7B, import_ckpt
from nemo.collections.multimodal.data.energon import EnergonMultiModalDataModule
from nemo.collections.multimodal.data.energon.config import MultiModalSampleConfig
from nemo.collections.multimodal.mimo.data.vqa import MimoVqaTaskEncoder
from nemo.collections.multimodal.mimo.model.config import MimoConfig
from nemo.collections.multimodal.mimo.model.model import MimoModel
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
from nemo.utils.exp_manager import TimingCallback


def main(args):

    # wandb.init(project=args.wandb_project, name=args.name)
    # Global and micro batch sizes
    gbs = args.gbs
    mbs = args.mbs
    seq_length = 8192
    stage = 'encoder_alignment'

    model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
    tokenizer = AutoTokenizer(model_id)
    special_tokens = [f"IMG_{i}" for i in range(8)]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    image_special_tokens = [f"IMG_{i}" for i in range(8)]
    image_special_token_indices = [tokenizer.tokenizer.convert_tokens_to_ids(f"IMG_{i}") for i in range(8)]

    processor = AutoProcessor.from_pretrained(model_id)
    image_processor = processor.image_processor

    multimodal_sample_config = MultiModalSampleConfig()
    task_encoder = MimoVqaTaskEncoder(
        tokenizer=tokenizer.tokenizer,
        image_processor=processor.image_processor,
        multimodal_sample_config=multimodal_sample_config,
    )

    data = EnergonMultiModalDataModule(
        path=args.data_path,
        tokenizer=tokenizer,
        image_processor=processor.image_processor,
        num_workers=32,
        seq_length=seq_length,
        micro_batch_size=mbs,
        global_batch_size=gbs,
        multimodal_sample_config=multimodal_sample_config,
        task_encoder=task_encoder,
    )

    mimo_config = MimoConfig(
        stage=stage,
        language_transformer_config=Llama2Config7B(seq_length=seq_length),
        vocab_size=tokenizer.vocab_size,
        image_special_token_indices=image_special_token_indices,
        image_special_tokens=image_special_tokens,
        freeze_language_model=args.freeze_language_model,
        language_model_path=args.language_model_path,
        vision_model_path=args.vision_model_path,
    )

    model = MimoModel(config=mimo_config, tokenizer=tokenizer)

    # Training strategy setup
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        # pipeline_dtype=torch.bfloat16,
        sequence_parallel=False,
    )

    # Checkpoint callback setup
    checkpoint_callback = nl.ModelCheckpoint(
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=2,
        every_n_train_steps=1000,
        dirpath=args.log_dir,
    )

    # Trainer setup
    trainer = nl.Trainer(
        num_nodes=args.num_nodes,
        devices=args.devices,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        callbacks=[checkpoint_callback, TimingCallback()],
        val_check_interval=50,
        limit_val_batches=gbs,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    # Logger setup
    nemo_logger = nl.NeMoLogger(
        log_dir=args.log_dir,
        name=args.name,
        wandb=WandbLogger(project=args.wandb_project, name=args.name) if args.wandb_project is not None else None,
    )

    # Auto resume setup
    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        resume_from_directory=args.log_dir,
        restore_config=nl.RestoreConfig(path=args.restore_path) if args.restore_path is not None else None,
    )

    # Optimizer and scheduler setup
    opt_config = OptimizerConfig(
        optimizer='adam',
        lr=args.lr,
        adam_beta1=0.9,
        adam_beta2=0.95,
        use_distributed_optimizer=True,
        bf16=True,
    )
    sched = CosineAnnealingScheduler(
        max_steps=trainer.max_steps,
        warmup_steps=150,
        constant_steps=0,
        min_lr=args.min_lr,
    )
    opt = MegatronOptimizerModule(opt_config, sched)

    # trainer.fit(model, data)
    llm.pretrain(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        optim=opt,
        resume=resume,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mimo Model Training Script")

    # Argument parsing
    parser.add_argument(
        "--log_dir", type=str, required=False, default="/root/.cache", help="Directory for logging and checkpoints"
    )
    parser.add_argument("--devices", type=int, required=False, default=1)
    parser.add_argument("--tp_size", type=int, required=False, default=1)
    parser.add_argument("--pp_size", type=int, required=False, default=1)
    parser.add_argument("--name", type=str, required=False, default="mimo_encoder_alignment_vqa")
    parser.add_argument("--wandb_project", type=str, required=False, default=None)
    parser.add_argument("--restore_path", type=str, required=False, default=None)
    parser.add_argument("--num_nodes", type=int, required=False, default=1)

    parser.add_argument("--gbs", type=int, required=False, default=64, help="Global batch size")
    parser.add_argument("--mbs", type=int, required=False, default=16, help="Micro batch size")
    parser.add_argument("--lr", type=float, required=False, default=2.0e-6, help="Learning rate")
    parser.add_argument("--min_lr", type=float, required=False, default=2.0e-7, help="Learning rate")

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--language_model_path", type=str, required=False, default=None)
    parser.add_argument("--max_steps", type=int, required=False, default=2500)
    parser.add_argument("--vision_model_path", type=str, required=False, default=None)
    parser.add_argument("--freeze_language_model", type=bool, required=False, default=True)
    args = parser.parse_args()
    main(args)
