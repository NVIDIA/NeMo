# import nemo_run as run
import pytorch_lightning as pl
import torch
from megatron.core.optimizer import OptimizerConfig
from pytorch_lightning.loggers import WandbLogger

from nemo import lightning as nl
from nemo.collections import llm


def logger() -> nl.NeMoLogger:
    # ckpt = None
    ckpt = nl.ModelCheckpoint(
        #save_best_model=True,
        save_last=True,
        every_n_train_steps=10,
        monitor="reduced_train_loss",
        save_top_k=1,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
    )

    wandb = None
    # wandb = WandbLogger(
    #         project="nemo2-squad",
    #         name='llama3-8b_lora-attn_test_api_wandb',
    #     )

    return nl.NeMoLogger(
        name="nemo2_peft",
        log_dir="/workspace/peftmerge/exp/peft_iomixin0",
        use_datetime_version=False,  # must be false if using auto resume
        ckpt=ckpt,
        wandb=wandb
    )

def trainer(devices=1) -> nl.Trainer:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=2,
    )

    return nl.Trainer(
        devices=2,
        max_steps=40,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        log_every_n_steps=1,
        limit_val_batches=2,
        val_check_interval=2,
        num_sanity_val_steps=0,
    )

def resume() -> nl.AutoResume:
    return nl.AutoResume(
        restore_config=nl.RestoreConfig(
            path="hf://meta-llama/Meta-Llama-3-8B",
        ),
        resume_if_exists=True,
        # resume_ignore_no_checkpoint=True,
    )

def llama3_8b() -> pl.LightningModule:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    return llm.LlamaModel(llm.Llama3Config8B(), tokenizer=tokenizer)


if __name__ == '__main__':
    llm.peft.merge_lora(
        model=llama3_8b(),
        trainer=trainer(),
        log=logger(),
        resume=resume(),
        output_path="/workspace/peftmerge/my_mergedump"
    )
