#!/usr/bin/python3
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
import os, re, sys
import tempfile
import torch
from functools import partial

import fiddle as fdl
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from nemo import lightning as nl
from nemo.collections import llm
from nemo.lightning.pytorch.callbacks import JitConfig, JitTransform
from transformers import AutoModelForCausalLM
from pathlib import Path

DATA_PATH = '/home/TestData/lite/hf_cache/squad/'


def make_squad_hf_dataset(data_path, tokenizer):
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""
        instruction = examples["context"]
        input = examples["question"]
        output = examples["answers"]['text']
        if isinstance(output, list):
            output = output[0]
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        ans = tokenizer(text)
        ans['labels'] = ans['input_ids']
        return ans

    tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)
    datamodule = llm.HFDatasetDataModule(data_path, split="train[:100]", pad_token_id=tokenizer.eos_token_id)

    datamodule.map(
        formatting_prompts_func,
        batched=False,
        batch_size=2,
        remove_columns=["id", "title", "context", "question", 'answers'],
    )

    return datamodule


def make_strategy(strategy, model, devices, num_nodes, adapter_only=False):
    if strategy == 'auto':
        return pl.strategies.SingleDeviceStrategy(
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


def logger(ckpt_folder) -> nl.NeMoLogger:
    ckpt = nl.ModelCheckpoint(
        save_last=True,
        every_n_train_steps=1,
        monitor="reduced_train_loss",
        save_top_k=1,
        save_on_train_epoch_end=True,
        save_optim_on_train_end=True,
    )

    return nl.NeMoLogger(
        name="nemo2_sft",
        log_dir=ckpt_folder,
        use_datetime_version=False,  # must be false if using auto resume
        ckpt=ckpt,
        wandb=None,
    )

def get_latest_checkpoint(base_dir):
    latest_checkpoint = None
    max_epoch = -1
    max_step = -1

    pattern = re.compile(r"[a-z0-9_]+--reduced_train_loss=([\d\.]+)-epoch=(\d+)-step=(\d+)-last")

    # Traverse through the base directory
    for root, dirs, _ in os.walk(base_dir):
        for dir_name in dirs:
            match = pattern.match(dir_name)
            if match:
                loss, epoch, step = map(float, match.groups())  # Convert to float/int
                epoch, step = int(epoch), int(step)

                # Update the latest checkpoint based on epoch and step
                if (epoch > max_epoch) or (epoch == max_epoch and step > max_step):
                    max_epoch = epoch
                    max_step = step
                    latest_checkpoint = os.path.join(root, dir_name)

    return Path(latest_checkpoint)



def verify_sft_checkpoint_structure(path):
    expected_files = set([
        'config.json',
        'generation_config.json',
        'model.safetensors',
        'special_tokens_map.json',
        'tokenizer.model',
        'tokenizer_config.json'
    ])
    ckpt_dir = Path(path)
    hf_weights = (ckpt_dir / "hf_weights")
    assert hf_weights.exists(), str(hf_weights)
    for file in hf_weights.glob('*'):
        assert file.name in expected_files, file
        expected_files.remove(file.name)
    assert len(expected_files) == 0

    assert (ckpt_dir / 'trainer.pt').exists()

    context_files = ['model.yaml', 'io.json']
    for file in context_files:
        assert (ckpt_dir / 'context' / file).exists()

def main():
    """Example script to run SFT with a HF transformers-instantiated model on squad."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--strategy', type=str, default='auto', choices=['auto', 'ddp', 'fsdp2'])
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--num-nodes', type=int, default=1)
    parser.add_argument('--accelerator', type=str, default='gpu', choices=['gpu'])
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--model-accelerator', type=str, default=None, choices=['te'])
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument("--fp8-autocast", action='store_true')
    parser.add_argument('--wandb-project', type=str, default=None)
    parser.add_argument('--ckpt-folder', type=str, default='/tmp/nemo_automodel_sft/')
    parser.add_argument('--use-torch-jit', action='store_true')
    parser.add_argument('--auto-resume', action='store_true')
    args = parser.parse_args()

    wandb = None
    if args.wandb_project is not None:
        model = '_'.join(args.model.split('/')[-2:])
        wandb = WandbLogger(
            project=args.wandb_project,
            name=f'{model}_dev{args.devices}_strat_{args.strategy}',
        )

    model_accelerator = None
    if args.model_accelerator == "te":
        from nemo.lightning.pytorch.accelerate.transformer_engine import te_accelerate

        model_accelerator = partial(te_accelerate, fp8_autocast=args.fp8_autocast)

    callbacks = []
    if args.use_torch_jit:
        jit_config = JitConfig(use_torch=True, torch_kwargs={'dynamic': False}, use_thunder=False)
        callbacks = [JitTransform(jit_config)]

    if args.auto_resume:
        callbacks.append(ValidateCheckpointRestoreCallback())

    class ZeroInitHFAutoModelForCausalLM(llm.HFAutoModelForCausalLM):
        def configure_module(self, *args, **kwargs):
            ans = super().configure_model(*args, **kwargs)
            for param in self.parameters():
                param.fill_(0)
            return ans

    model_cls = ZeroInitHFAutoModelForCausalLM if args.auto_resume else llm.HFAutoModelForCausalLM
    model = model_cls(model_name=args.model, model_accelerator=model_accelerator)

    strategy = make_strategy(args.strategy, model, args.devices, args.num_nodes, False)

    resume = (
        nl.AutoResume(
            resume_if_exists=True,
            resume_ignore_no_checkpoint=False,
        )
        if args.auto_resume
        else None
    )
    args.max_steps += int(resume is not None)
    trainer = nl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        strategy=strategy,
        log_every_n_steps=1,
        limit_val_batches=0.0,
        num_sanity_val_steps=0,
        accumulate_grad_batches=1,
        gradient_clip_val=args.grad_clip,
        use_distributed_sampler=False,
        logger=wandb,
        callbacks=callbacks,
        precision="bf16",
    )

    llm.api.finetune(
        model=model,
        data=make_squad_hf_dataset(DATA_PATH, llm.HFAutoModelForCausalLM.configure_tokenizer(args.model)),
        trainer=trainer,
        optim=fdl.build(llm.adam.pytorch_adam_with_flat_lr(lr=1e-5)),
        log=logger(args.ckpt_folder),
        resume=resume,
    )

    del model
    del trainer

    path = get_latest_checkpoint(args.ckpt_folder)
    print('paht= ' + str(path))
    verify_sft_checkpoint_structure(path)

    ans = AutoModelForCausalLM.from_pretrained(path / "hf_weights", output_loading_info=True)
    assert len(ans[1]['missing_keys']) == 0, ("NOT LOADABLE #1", ans)
    assert len(ans[1]['mismatched_keys']) == 0, ("NOT LOADABLE #2", ans)


if __name__ == '__main__':
    main()
