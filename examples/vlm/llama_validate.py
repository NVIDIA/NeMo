from nemo import lightning as nl
from nemo.collections import llm, vlm
import pytorch_lightning as pl
import argparse


# @run.factory
def trainer(devices) -> nl.Trainer:
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=devices,
        setup_optimizers=False,
        # save_ckpt_format='zarr',
    )

    return nl.Trainer(
        devices=devices,
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
def squad() -> pl.LightningDataModule:
    return llm.FineTuningDataModule(dataset_root="/lustre/fsw/coreai_dlalgo_llm/nemo_home/datasets/samples/squad_1",
                                    seq_length=2048, micro_batch_size=1,
                                    global_batch_size=128, num_workers=0)


# @run.factory
def llama32() -> pl.LightningModule:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-11B-Vision")
    # set language_model_config or vision_model_config to None to load only one part
    return vlm.MLlamaModel(
        vlm.MLlamaModelConfig(
            language_model_config=vlm.CrossAttentionTextModelConfig8B(rotary_interleaved=True, apply_rope_fusion=False),
            vision_model_config=vlm.CrossAttentionVisionModelConfig(num_layers=32, hidden_size=1280,
                                                                    num_attention_heads=16, vision_chunk_size=448,
                                                                    vision_max_num_chunks=4, ),
        ),
        tokenizer=tokenizer)


# @run.factory
def resume() -> nl.AutoResume:
    return nl.AutoResume(
        restore_config=nl.RestoreConfig(
            # path="pytorch:///lustre/fsw/coreai_dlalgo_llm/aot/checkpoints/evian3/evian3-11b-vision-final_vv1/consolidated.pth",
            path="hf://meta-llama/Llama-3.2-11B-Vision",
        ),
        resume_if_exists=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trainer Configuration")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use for training")

    args = parser.parse_args()

    llm.validate(
        model=llama32(),
        data=squad(),
        trainer=trainer(devices=args.devices),
        log=logger(),
        resume=resume(),
    )
