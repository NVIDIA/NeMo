## NOTE: This script is present for github-actions testing only.
## There are no guarantees that this script is up-to-date with latest NeMo.

import argparse

from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.api import finetune
from nemo.lightning import NeMoLogger
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule


def get_args():
    parser = argparse.ArgumentParser(description='Train Llama2-7B with SFT using the Squad dataset.')
    parser.add_argument('--devices', type=int, help="Number of devices to use for training")
    parser.add_argument('--max-steps', type=int, help="Number of steps to train for")
    parser.add_argument('--experiment-dir', type=str, help="directory to write results and checkpoints to")
    parser.add_argument('--gbs', type=int, default=128, help="Global batch size.")
    parser.add_argument('--mbs', type=int, default=1, help="Micro batch size.")
    parser.add_argument('--hf-ckpt-path', type=str, default="hf://meta-llama/Llama-2-7b-hf", help="HF checkpoint path to import.")


    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    # Configure the optimizer.
    optim = nl.MegatronOptimizerModule(
        config=OptimizerConfig(
            optimizer="adam",
            lr=5e-6,
            use_distributed_optimizer=False,
            bf16=True,
        ),
    )

    lora = llm.peft.LoRA(
        target_modules=['linear_qkv', 'linear_proj'],
        dim=32,
    )

    # Configure the model.
    model = llm.LlamaModel(
        config=llm.Llama2Config7B(
            masked_softmax_fusion=False,
        ),
        model_transform=lora,
        optim=optim
    )
    # Import the checkpoint.
    ckpt_path = model.import_ckpt(args.hf_ckpt_path)

    # Configure the dataset.
    data = llm.SquadDataModule(
        seq_length=2048,
        micro_batch_size=args.mbs,
        global_batch_size=args.gbs,
        tokenizer=model.tokenizer,
        num_workers=int(args.devices * 2),
    )

    # Setup the trainer.
    trainer = nl.Trainer(
        devices=args.devices,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=nl.MegatronStrategy(
            tensor_model_parallel_size=args.devices,
            ckpt_include_optimizer=False,
        ),
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
        limit_val_batches=0.0,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        callbacks=[lora],
    )

    # Setup NeMoLogger.
    nemo_logger = NeMoLogger(
        dir=args.experiment_dir,
    )

    # Run finetuning.
    finetune(
        model=model,
        data=data,
        trainer=trainer,
        log=nemo_logger,
        optim=optim,
    )
