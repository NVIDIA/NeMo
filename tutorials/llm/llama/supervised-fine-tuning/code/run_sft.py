## script to do SFT with NeMo 2.0
import nemo_run as run
from nemo import lightning as nl
from nemo.collections import llm
import torch
import pytorch_lightning as pl
from pathlib import Path
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.optimizer import OptimizerConfig
from nemo.collections.llm import Llama2Config7B
import wandb
from lightning.pytorch.loggers import WandbLogger
from typing import List, Optional
from nemo.lightning.io.mixin import IOMixin
from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule
from verilog_data_module import VerilogDataModule

# configure custom dataset
def verilog() -> run.Config[pl.LightningDataModule]:
    return run.Config(VerilogDataModule, seq_length=1024, micro_batch_size=2, global_batch_size=8, num_workers=8)

# configure trainer class similar to pytorch lightning trainer
def trainer() -> run.Config[nl.Trainer]:
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=2
    )
    trainer = run.Config(
        nl.Trainer,
        devices=2,
        max_steps=200,
        accelerator="gpu",
        strategy=strategy,
        plugins=bf16_mixed(),
        log_every_n_steps=40,
        limit_val_batches=2,
        val_check_interval=20,
        num_sanity_val_steps=0,
    )
    return trainer

# configure the logger
def logger() -> run.Config[nl.NeMoLogger]:
    ckpt = run.Config(
        nl.ModelCheckpoint,
        save_last=True,
        every_n_train_steps=40,
        monitor="val_loss",
        save_top_k=1,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
    )

    ## this is where hthe
    return run.Config(
        nl.NeMoLogger,
        name="sft_log",
        log_dir="//workspace",
        use_datetime_version=False,
        ckpt=ckpt,
        wandb=None,
    )

# configre the optimizer, adam with cosine annealing
def adam_with_cosine_annealing() -> run.Config[nl.OptimizerModule]:
    opt_cfg = run.Config(
        OptimizerConfig,
        optimizer="adam",
        lr=5e-5,
        adam_beta2=0.98,
        use_distributed_optimizer=True,
        clip_grad=1.0,
        bf16=True,
    )
    return run.Config(
        nl.MegatronOptimizerModule,
        config=opt_cfg
    )

# configure the base model
def llama2_7b() -> run.Config[pl.LightningModule]:
    return run.Config(llm.LlamaModel, config=run.Config(llm.Llama2Config7B))

# configure auto resume
def resume() -> run.Config[nl.AutoResume]:
    return run.Config(
        nl.AutoResume,
        restore_config=run.Config(nl.RestoreConfig,
        ## default path to save converted hf model
            path="/root/.cache/nemo/models/Llama-2-7b-hf"
        ),
        # requires completely saved checkpoint to resume from
        resume_if_exists=False,
    )

# with all above components created, call NeMo2.0 finetune API
def configure_finetuning_recipe():
    return run.Partial(
        llm.finetune,
        model=llama2_7b(),        
        trainer=trainer(),
        data=verilog(),
        log=logger(),
        optim=adam_with_cosine_annealing(),
        resume=resume(),
    )


def local_executor_torchrun(nodes: int = 1, devices: int = 2) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)
    return executor


def main():
    print("preprocess data!")
    verilog = VerilogDataModule()
    verilog_data = verilog._download_data()
    verilog._preprocess_and_split_data(verilog_data)
    print("running supervised fine tuning!")
    run.run(configure_finetuning_recipe(), executor=local_executor_torchrun())

if __name__ == "__main__":
    main()