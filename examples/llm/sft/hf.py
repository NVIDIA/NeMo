import fiddle as fdl
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from nemo import lightning as nl
from nemo.collections import llm


class SquadDataModuleWithMbs(llm.SquadDataModule):
    def _create_dataloader(self, dataset, **kwargs) -> DataLoader:
        from nemo.lightning.data import add_megatron_sampler

        kwargs1 = {
            'consumed_samples': 0,
            'dataloader_type': 'single',
            'drop_last': True,
            'pad_samples_to_global_batch_size': False,
        }
        return add_megatron_sampler(
            DataLoader(
                dataset,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                collate_fn=dataset.collate_fn,
                **kwargs,
            ),
            self.micro_batch_size,
            self.global_batch_size,
            **kwargs1,
        )


def squad(tokenizer) -> pl.LightningDataModule:
    return SquadDataModuleWithMbs(
        tokenizer=tokenizer,
        seq_length=2048,
        micro_batch_size=2,
        global_batch_size=128,
        num_workers=0,
        sanity_check_dist_workers=False,
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--strategy', type=str, default='auto', choices=['auto', 'ddp', 'fsdp'])
    parser.add_argument('--devices', default=1)
    parser.add_argument('--accelerator', default='gpu', choices=['gpu'])
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--wandb-project', type=str, default=None)
    args = parser.parse_args()

    wandb = None
    if args.wandb_project is not None:
        model = '_'.join(args.model.split('/')[-2:])
        wandb = WandbLogger(
            project=args.wandb_project,
            name=f'{model}_dev{args.devices}_strat_{args.strategy}',
        )
    grad_clip = 0.5
    if args.strategy == 'fsdp':
        # See: https://github.com/Lightning-AI/pytorch-lightning/blob/8ad3e29816a63d8ce5c00ac104b14729a4176f4f/src/lightning/pytorch/plugins/precision/fsdp.py#L81
        grad_clip = None
    use_dist_samp = False

    llm.api.finetune(
        model=llm.HfAutoModelForCausalLM(args.model),
        data=squad(llm.HfAutoModelForCausalLM.configure_tokenizer(args.model)),
        trainer=nl.Trainer(
            devices=args.devices,
            max_steps=args.max_steps,
            accelerator=args.accelerator,
            strategy=args.strategy,
            log_every_n_steps=1,
            limit_val_batches=0.0,
            num_sanity_val_steps=0,
            accumulate_grad_batches=10,
            gradient_clip_val=grad_clip,
            use_distributed_sampler=use_dist_samp,
            logger=wandb,
        ),
        optim=fdl.build(llm.adam.pytorch_adam_with_flat_lr(max_lr=1e-5, clip_grad=0.5)),
        log=None,
    )
