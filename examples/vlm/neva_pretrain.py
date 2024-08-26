## NOTE: This script is present for github-actions testing only.
## There are no guarantees that this script is up-to-date with latest NeMo.

import argparse

from megatron.core.optimizer import OptimizerConfig
from nemo.lightning.pytorch.optim import (
    CosineAnnealingScheduler,
    MegatronOptimizerModule,
    OptimizerModule,
)
from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.api import train
from nemo.collections import vlm
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule

if __name__ == "__main__":
    gbs = 256
    mbs = 4
    seq_length = 2048
    log_dir = "/opt/nemo_runs"

    # data module
    # data = vlm.MockDataModule(
    #     seq_length=seq_length,
    #     global_batch_size=gbs,
    #     micro_batch_size=mbs,
    #     tokenizer=None,
    #     image_processor=None,
    #     num_workers=0,
    # )

    from nemo.collections.vlm.neva.data.config import ImageDataConfig, DataConfig

    data_config = ImageDataConfig(
        image_folder="/lustre/fsw/coreai_dlalgo_genai/datasets/LLaVA-Pretrain-LCS-558K/images",
        conv_template="plain",
    )
    data = vlm.NevaLazyDataModule(
        paths="/lustre/fsw/coreai_dlalgo_genai/datasets/LLaVA-Pretrain-LCS-558K/blip_laion_cc_sbu_558k.json",
        data_config=data_config,
        seq_length=seq_length,
        global_batch_size=gbs,
        micro_batch_size=mbs,
        tokenizer=None,
        image_processor=None,
        num_workers=0,
    )

    language_transformer_config = llm.Llama2Config7B()
    # vision_transformer_config = vlm.CLIPViTConfig(num_layers=2, hidden_size=1024, num_attention_heads=4)
    from nemo.collections.vlm.neva.model.base import HFCLIPVisionConfig
    vision_transformer_config = HFCLIPVisionConfig(pretrained_model_name_or_path="openai/clip-vit-large-patch14-336")
    vision_projection_config = vlm.MultimodalProjectorConfig(input_size=1024, hidden_size=4096)

    neva_config = vlm.NevaConfig(
        language_transformer_config=language_transformer_config,
        vision_transformer_config=vision_transformer_config,
        vision_projection_config=vision_projection_config,
        language_model_from_pretrained="/root/.cache/nemo/models/lmsys/vicuna-7b-v1.5",
    )

    model = vlm.NevaModel(neva_config, tokenizer=data.tokenizer)

    strategy = nl.MegatronStrategy(tensor_model_parallel_size=1)

    checkpoint_callback = nl.ModelCheckpoint(
        save_best_model=True,
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=2,
        every_n_train_steps=10,
        enable_nemo_ckpt_io=False,
        dirpath=log_dir,
    )

    trainer = nl.Trainer(
        devices=1,
        max_steps=2170,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        callbacks=[checkpoint_callback],
        val_check_interval=1000,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )

    nemo_logger = nl.NeMoLogger(
            dir=log_dir,
            name="neva_test",
            wandb=WandbLogger(project="neva_demo")
        )
    nemo_logger.setup(
        trainer,
        resume_if_exists=True,
    )

    resume = nl.AutoResume(resume_if_exists=True, resume_ignore_no_checkpoint=True, dirpath=log_dir)
    resume.setup(trainer, model)


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
