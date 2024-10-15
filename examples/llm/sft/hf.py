from nemo import lightning as nl
from nemo.collections import llm
from megatron.core.optimizer import OptimizerConfig
import torch
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

class SquadDataModuleWithMbs(llm.SquadDataModule):
    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        from nemo.lightning.data import add_megatron_sampler
        kwargs1 = {'consumed_samples': 0, 'dataloader_type': 'single',
                  'drop_last': True, 'pad_samples_to_global_batch_size': False}
        return add_megatron_sampler(DataLoader(
                dataset,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                collate_fn=dataset.collate_fn,
                **kwargs,
            ), self.micro_batch_size, self.global_batch_size, **kwargs1)


def squad(tokenizer) -> pl.LightningDataModule:
    return SquadDataModuleWithMbs(
        tokenizer=tokenizer, seq_length=2048, micro_batch_size=2, global_batch_size=8,
        num_workers=0, sanity_check_dist_workers=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B')
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
