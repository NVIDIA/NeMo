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

import nemo_run as run
from lightning.pytorch.loggers import WandbLogger
from nemo.collections import llm

DATA_PATH = '/home/TestData/lite/hf_cache/squad/'
DATA_PATH = '/lustre/fsw/coreai_dlalgo_llm/huiyingl/liteci/squad/'


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
    datamodule = llm.HFDatasetDataModule(data_path, split="train", pad_token_id=tokenizer.eos_token_id)

    datamodule.map(
        formatting_prompts_func,
        batched=False,
        batch_size=2,
        remove_columns=["id", "title", "context", "question", 'answers'],
    )

    return datamodule


def local_executor_torchrun(nodes: int = 1, devices: int = 2) -> run.LocalExecutor:
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
        "NVTE_FUSED_ATTN": "0",
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor

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
    tokenizer = llm.HFAutoModelForCausalLM.configure_tokenizer(args.model)

    recipe = llm.hf_auto_model_for_causal_lm.finetune_recipe(
        model_name=args.model,
        name="gemma_finetune",
        num_nodes=1,
        num_gpus_per_node=1,
        peft_scheme='lora',
    )
    recipe.trainer.val_check_interval = 50

    executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)
    run.run(recipe, executor=executor)

