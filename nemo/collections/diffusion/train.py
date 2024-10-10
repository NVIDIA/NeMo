# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import nemo_run as run
import pytorch_lightning as pl
import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import WandbLogger

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.diffusion.data.diffusion_energon_datamodule import DiffusionDataModule
from nemo.collections.diffusion.data.diffusion_taskencoder import BasicDiffusionTaskEncoder
from nemo.collections.diffusion.models.model import (
    DiT5BConfig,
    DiT30BConfig,
    DiTConfig,
    DiTLConfig,
    DiTModel,
    DiTXLConfig,
)
from nemo.lightning.pytorch.callbacks import ModelCheckpoint, PreemptionCallback
from nemo.lightning.pytorch.callbacks.model_transform import ModelTransform
from nemo.lightning.pytorch.strategies.utils import RestoreConfig


@run.cli.factory
@run.autoconvert
def multimodal_datamodule() -> pl.LightningDataModule:
    data_module = DiffusionDataModule(
        seq_length=2048,
        task_encoder=run.Config(BasicDiffusionTaskEncoder, seq_length=2048),
        micro_batch_size=1,
        global_batch_size=32,
    )
    return data_module


@run.cli.factory
@run.autoconvert
def peft(args) -> ModelTransform:
    return llm.peft.LoRA(
        target_modules=['linear_qkv', 'linear_proj'],  # , 'linear_fc1', 'linear_fc2'],
        dim=args.lora_dim,
    )


@run.cli.factory(target=llm.train)
def pretrain() -> run.Partial:
    return run.Partial(
        llm.train,
        model=run.Config(
            DiTModel,
            config=run.Config(DiTConfig),
        ),
        data=multimodal_datamodule(),
        trainer=run.Config(
            nl.Trainer,
            devices='auto',
            num_nodes=int(os.environ.get('SLURM_NNODES', 1)),
            accelerator="gpu",
            strategy=run.Config(
                nl.MegatronStrategy,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                sequence_parallel=False,
                pipeline_dtype=torch.bfloat16,
                ddp=run.Config(DistributedDataParallelConfig, check_for_nan_in_grad=True),
            ),
            plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
            num_sanity_val_steps=0,
            limit_val_batches=1,
            val_check_interval=1000,
            max_epochs=10000,
            log_every_n_steps=1,
            callbacks=[
                run.Config(
                    ModelCheckpoint,
                    monitor='reduced_train_loss',
                    filename='{epoch}-{step}',
                    every_n_train_steps=1000,
                    save_top_k=-1,
                ),
                run.Config(PreemptionCallback),
            ],
        ),
        log=nl.NeMoLogger(wandb=(WandbLogger(project='dit') if "WANDB_API_KEY" in os.environ else None)),
        optim=run.Config(
            nl.MegatronOptimizerModule,
            config=run.Config(
                OptimizerConfig,
                lr=1e-4,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=True,
                weight_decay=0,
            ),
        ),
        tokenizer=None,
        resume=run.Config(
            nl.AutoResume,
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
            resume_past_end=True,
        ),
        model_transform=None,
    )


@run.cli.factory(target=llm.train)
def pretrain_xl() -> run.Partial:
    recipe = pretrain()
    recipe.model.config = run.Config(DiTXLConfig)
    return recipe


@run.cli.factory(target=llm.train)
def pretrain_l() -> run.Partial:
    recipe = pretrain()
    recipe.model.config = run.Config(DiTLConfig)
    return recipe


@run.cli.factory(target=llm.train)
def pretrain_5b() -> run.Partial:
    recipe = pretrain()
    recipe.model.config = run.Config(DiT5BConfig)
    recipe.data.global_batch_size = 4608
    recipe.data.micro_batch_size = 9
    recipe.data.num_workers = 8
    recipe.data.use_train_split_for_val = True
    recipe.data.seq_length = 260
    recipe.data.task_encoder.seq_length = 260
    recipe.trainer.val_check_interval = 1000
    recipe.log.log_dir = 'nemo_experiments/dit5b_256px'
    recipe.model.config.sigma_data = 1.0
    recipe.optim.lr_scheduler = run.Config(nl.lr_scheduler.WarmupHoldPolicyScheduler, warmup_steps=100, hold_steps=1e9)

    return recipe


@run.cli.factory(target=llm.train)
def pretrain_30b() -> run.Partial:
    recipe = pretrain()
    recipe.model.config = run.Config(DiT30BConfig)
    recipe.data.global_batch_size = 9216
    recipe.data.micro_batch_size = 1
    recipe.data.num_workers = 8
    recipe.data.use_train_split_for_val = True
    recipe.data.seq_length = 260
    recipe.data.task_encoder.seq_length = 260
    recipe.trainer.val_check_interval = 1000
    recipe.log.log_dir = 'nemo_experiments/dit30b'
    return recipe


@run.cli.factory(target=llm.train)
def dreambooth() -> run.Partial:
    recipe = pretrain()
    recipe.optim.config.lr = 1e-6
    recipe.data = multimodal_datamodule()
    recipe.model.config = run.Config(DiTConfig)

    recipe.trainer.max_steps = 1000
    recipe.trainer.strategy.tensor_model_parallel_size = 8
    recipe.trainer.strategy.sequence_parallel = True

    recipe.resume.restore_config = run.Config(RestoreConfig)
    recipe.resume.resume_if_exists = False

    return recipe


if __name__ == "__main__":
    run.cli.main(llm.train, default_factory=dreambooth)
