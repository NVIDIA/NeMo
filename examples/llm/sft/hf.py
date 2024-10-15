from nemo import lightning as nl
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig
import torch
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset



def squad(tokenizer) -> pl.LightningDataModule:
    return llm.SquadDataModule(
        tokenizer=tokenizer, seq_length=2048, micro_batch_size=2, global_batch_size=8,
        num_workers=0, sanity_check_dist_workers=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/mnt/4tb/nemo_lite/llama3_1b')
    parser.add_argument('--strategy', type=str, default='auto', choices=['auto', 'ddp', 'fsdp'])
    parser.add_argument('--devices', default=1)
    parser.add_argument('--accelerator', default='gpu', choices=['gpu'])
    parser.add_argument('--max-steps', type=int, default=100)
    args = parser.parse_args()

    model = llm.HfAutoModel(args.model, {"lr": 1e-5})
    llm.api.finetune(
        model=model,
        data=squad(model.tokenizer),
        trainer=nl.Trainer(
            devices=args.devices,
            max_steps=args.max_steps,
            accelerator=args.accelerator,
            strategy=args.strategy,
            log_every_n_steps=1,
            limit_val_batches=0.0,
            num_sanity_val_steps=0,
            accumulate_grad_batches=10,
            gradient_clip_val=0.5,
        ),
        log=None,
    )
