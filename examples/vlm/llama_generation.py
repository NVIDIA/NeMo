import nemo_run as run
from nemo import lightning as nl
from nemo.collections import llm, vlm
from megatron.core.optimizer import OptimizerConfig
import torch
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl



# @run.factory
def trainer(devices=1) -> nl.Trainer:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=4,
    )

    return nl.Trainer(
        devices=4,
        max_steps=2,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        log_every_n_steps=1,
        limit_val_batches=2,
        val_check_interval=2,
        num_sanity_val_steps=0,
    )


# @run.factory
def logger() -> nl.NeMoLogger:
    # ckpt = None
    ckpt = nl.ModelCheckpoint(
        save_last=True,
        monitor="reduced_train_loss",
        save_top_k=1,
        save_on_train_epoch_end=True,
    )

    wandb = None
    # wandb = WandbLogger(
    #         project="nemo2-squad",
    #         name='llama3-8b_lora-attn_test_api_wandb',
    #     )

    return nl.NeMoLogger(
        name="evian3_testload",
        dir="/chcui/exp/evian3",
        use_datetime_version=False,  # must be false if using auto resume
        ckpt=ckpt,
        wandb=wandb
    )


# @run.factory
def adam_with_cosine_annealing() -> nl.OptimizerModule:
    return nl.MegatronOptimizerModule(
        config=OptimizerConfig(
            optimizer="adam",
            lr=0.0001,
            adam_beta2=0.98,
            use_distributed_optimizer=True,
            clip_grad=1.0,
            bf16=True,
            # params_dtype=torch.bfloat16,
        ),
        # lr_scheduler=nl.lr_scheduler.CosineAnnealingScheduler(max_steps=100),
    )

# @run.factory
# def lora() -> nl.pytorch.callbacks.PEFT:
#     return llm.peft.LoRA()
    # return llm.peft.LoRA(
    #     target_modules=['linear_qkv', 'linear_proj'], #, 'linear_fc1', 'linear_fc2'],
    #     dim=32,
    # )

# @run.factory
def squad() -> pl.LightningDataModule:
    return llm.FineTuningDataModule(dataset_root="/lustre/fsw/coreai_dlalgo_llm/nemo_home/datasets/samples/squad_1",
                                    seq_length=2048, micro_batch_size=1,
                                    global_batch_size=128, num_workers=0)


# @run.factory
def llama32() -> pl.LightningModule:
    # from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
    # tokenizer = get_nmt_tokenizer(
    #         library="sentencepiece",
    #         # model_name=self.model_config.tokenizer_name,
    #         tokenizer_model="/lustre/fsw/coreai_dlalgo_llm/aot/checkpoints/evian3/evian3-11b-vision-early_vv1/tokenizer.model",
    #         use_fast=True,
    #     )
    from nemo.collections.vlm.llama.model.base import LlamaCrossAttentionModel, CrossAttentionVisionModelConfig, LlamaCrossAttentionModelConfig, CrossAttentionTextModelConfig8B
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    # tokenizer = None
    return LlamaCrossAttentionModel(
        LlamaCrossAttentionModelConfig(
            language_model_config=CrossAttentionTextModelConfig8B(),
            vision_model_config=CrossAttentionVisionModelConfig(
                num_layers=32, hidden_size=1280, num_attention_heads=16, vision_chunk_size=448, vision_max_num_chunks=4,
            ),
        ),
        tokenizer=tokenizer)

# @run.factory
def resume() -> nl.AutoResume:
    return nl.AutoResume(
        restore_config=nl.RestoreConfig(
            path="pytorch:///lustre/fsw/coreai_dlalgo_llm/aot/checkpoints/evian3/evian3-11b-vision-early_vv1/consolidated.pth",
        ),
        resume_if_exists=True,
        # resume_ignore_no_checkpoint=True,
    )

if __name__ == '__main__':
    llm.finetune(
        model=llama32(),
        data=squad(),
        trainer=trainer(),
        peft=None,
        log=logger(),
        optim=adam_with_cosine_annealing(),
        resume=resume(),
    )