# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import fiddle as fdl
import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import WandbLogger

from nemo import lightning as nl
from nemo.collections import llm, vlm

DATA_PATH = "/home/TestData/vlm/rdr-items"


def mk_hf_vlm_dataset(processor, mbs, gbs):
    skipped_tokens = vlm.HFAutoModelForImageTextToText.extract_skipped_token_ids(processor)

    def collate_fn(examples, processor):
        def fmt(sample):
            instruction = "Describe accurately the given image."
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": instruction}, {"type": "image", "image": sample["image"]}],
                },
                {"role": "assistant", "content": [{"type": "text", "text": sample["text"]}]},
            ]
            return {"conversation": conversation, "images": [sample['image']]}

        text = []
        images = []
        for example in map(fmt, examples):
            text.append(
                processor.apply_chat_template(
                    example["conversation"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            images += example['images']

        # Tokenize the text and process the images
        batch = processor(
            text=text,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)

        labels = batch["input_ids"].clone()
        labels[torch.isin(labels, skipped_tokens)] = -100
        batch["labels"] = labels
        return batch

    return vlm.HFDatasetDataModule(
        DATA_PATH,
        split="train[:10]",
        micro_batch_size=mbs,
        global_batch_size=gbs,
        collate_fn=lambda x: collate_fn(x, processor=processor),
    )


def make_strategy(strategy, model, devices, num_nodes, adapter_only=False):
    if strategy == 'auto':
        return pl.strategies.SingleDeviceStrategy(
            device='cuda:0',
            checkpoint_io=model.make_checkpoint_io(adapter_only=adapter_only),
        )
    elif strategy == 'ddp':
        return pl.strategies.DDPStrategy(
            checkpoint_io=model.make_checkpoint_io(adapter_only=adapter_only),
        )
    elif strategy == 'fsdp2':
        return nl.FSDP2Strategy(
            data_parallel_size=devices * num_nodes,
            tensor_parallel_size=1,
            checkpoint_io=model.make_checkpoint_io(adapter_only=adapter_only),
        )
    else:
        raise NotImplementedError("Encountered unknown strategy")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2-VL-2B-Instruct')
    parser.add_argument('--strategy', type=str, default='auto', choices=['auto', 'ddp', 'fsdp2'])
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--mbs', default=1)
    parser.add_argument('--gbs', default=1)
    parser.add_argument('--accelerator', default='gpu', choices=['gpu'])
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--wandb-project', type=str, default=None)
    parser.add_argument('--disable-ckpt', action='store_false')
    parser.add_argument('--use-4bit', help="Load model in 4bit", action="store_true")
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
    processor = vlm.HFAutoModelForImageTextToText.configure_processor(args.model)
    model = vlm.HFAutoModelForImageTextToText(args.model, load_in_4bit=args.use_4bit)
    llm.api.finetune(
        model=model,
        data=mk_hf_vlm_dataset(processor, args.mbs, args.gbs),
        trainer=nl.Trainer(
            devices=args.devices,
            max_steps=args.max_steps,
            accelerator=args.accelerator,
            strategy=make_strategy(args.strategy, model, args.devices, 1, adapter_only=True),
            log_every_n_steps=1,
            limit_val_batches=0.0,
            num_sanity_val_steps=0,
            accumulate_grad_batches=2,
            gradient_clip_val=grad_clip,
            use_distributed_sampler=use_dist_samp,
            logger=wandb,
            enable_checkpointing=args.disable_ckpt,
            precision='bf16',
        ),
        optim=fdl.build(llm.adam.pytorch_adam_with_flat_lr(lr=1e-5)),
        log=None,
        peft=llm.peft.LoRA(
            target_modules=['*_proj'],
            dim=16,
            lora_dtype=torch.bfloat16 if args.use_4bit else None,
        ),
    )
