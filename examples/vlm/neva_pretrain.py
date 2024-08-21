## NOTE: This script is present for github-actions testing only.
## There are no guarantees that this script is up-to-date with latest NeMo.

import argparse

from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import TensorBoardLogger

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.api import train
from nemo.collections import vlm
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.collections.llm.recipes.log.default import default_log, default_resume
gbs = 8
mbs = 2
seq_length = 1024

log_dir = "/opt/nemo_runs"
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

callbacks = [checkpoint_callback]

trainer = nl.Trainer(
    devices=1,
    max_steps=1000,
    accelerator="gpu",
    strategy=strategy,
    callbacks=callbacks,
    plugins=nl.MegatronMixedPrecision(precision="bf16-mixed", ),
    val_check_interval=1000,
    log_every_n_steps=1,
)

opt_config = OptimizerConfig(
    optimizer='adam',
    lr=6e-4,
    min_lr=6e-5,
    use_distributed_optimizer=False,
    bf16=True,
)
opt = MegatronOptimizerModule(config=opt_config)

nemo_logger = nl.NeMoLogger(
        dir=log_dir,
        # wandb=WandbLogger(entity="nvidia", project="nemo-ux-demo")
    )

data = vlm.MockDataModule(
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
    language_model_from_pretrained="/root/.cache/nemo/models/lmsys/vicuna-7b-v1.5/",
)

model = vlm.NevaModel(neva_config, tokenizer=data.tokenizer)
resume = nl.AutoResume(resume_if_exists=True, resume_ignore_no_checkpoint=True, dirpath=log_dir)

train(
    model=model,
    data=data,
    trainer=trainer,
    log=nemo_logger,
    tokenizer='data',
    optim=opt,
    resume=resume,
)
