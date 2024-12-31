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

import nemo.lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.collections.llm.gpt.data.hf_dataset import SquadHFDataModule


DATA_PATH = '/lustre/fsw/coreai_dlalgo_llm/boxiangw/squad'

import torch
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)


def parallelize(model, device_mesh: DeviceMesh):
    """Apply parallelisms and activation checkpointing to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.

    """

    dp_mesh = device_mesh["data_parallel"]
    tp_mesh = device_mesh["tensor_parallel"]

    if tp_mesh.size() > 1:
        # 1. Parallelize the first embedding and the last linear proj layer
        # 2. Parallelize the root norm layer over the sequence dim
        # 3. Shard the first transformer block's inputs

        # Parallelize the first embedding and the last linear out projection
        plan = {
            "tok_embeddings": RowwiseParallel(input_layouts=Replicate()),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                # Optional: Shard the output along the class dimension to compute the loss in parallel.
                # See `loss_parallel` in `train.py`
                output_layouts=Shard(-1),
                use_local_output=False,
            ),
            "norm": SequenceParallel(),
            "layers.0": PrepareModuleInput(
                input_layouts=(Replicate(), None),
                desired_input_layouts=(Shard(1), None),
                use_local_output=True,
            ),
        }
        model = parallelize_module(model, tp_mesh, plan)

        # Parallelize each transformer block
        for transformer_block in model.layers.values():
            plan = {
                "attention": PrepareModuleInput(
                    input_layouts=(Shard(1), None),
                    desired_input_layouts=(Replicate(), None),
                ),
                "attention.wq": ColwiseParallel(),
                "attention.wk": ColwiseParallel(),
                "attention.wv": ColwiseParallel(),
                "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
                "attention_norm": SequenceParallel(),
                "feed_forward": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "feed_forward.w1": ColwiseParallel(),
                "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
                "feed_forward.w3": ColwiseParallel(),
                "ffn_norm": SequenceParallel(),
            }

            # Adjust attention module to use the local number of heads
            attn_layer = transformer_block.attention
            attn_layer.n_heads = attn_layer.n_heads // tp_mesh.size()
            attn_layer.n_kv_heads = attn_layer.n_kv_heads // tp_mesh.size()

            # Apply the plan for the current transformer block
            parallelize_module(transformer_block, tp_mesh, plan)

    if dp_mesh.size() > 1:
        assert dp_mesh.ndim == 1  # Hybrid-sharding not supported

        # NOTE: Currently, the user is required to manually handle precision settings such as the `mp_policy` here
        # because the model parallel strategy does not respect all settings of `Fabric(precision=...)` at the moment.
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

        fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
        for layer_id, transformer_block in model.layers.items():
            # Apply activation checkpointing
            transformer_block = checkpoint_wrapper(transformer_block)
            # As an optimization, do not reshard after forward for the last
            # transformer block since FSDP would prefetch it immediately
            reshard_after_forward = int(layer_id) < len(model.layers) - 1
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
            model.layers[layer_id] = transformer_block
        model = fully_shard(model, **fsdp_config)

    return model


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
    parser.add_argument('--devices', default=8)
    parser.add_argument('--accelerator', default='gpu', choices=['gpu'])
    parser.add_argument('--max-steps', type=int, default=1000)
    args = parser.parse_args()

    recipe = llm.hf_auto_model_for_causal_lm.finetune_recipe(
        model_name=args.model,
        name="sft",
        num_nodes=1,
        num_gpus_per_node=args.devices,
        peft_scheme='none',
        max_steps=args.max_steps,
    )
    recipe.trainer.val_check_interval = 50

    tokenizer = llm.HFAutoModelForCausalLM.configure_tokenizer(args.model)
    recipe.data = run.Config(
        SquadHFDataModule,
        path_or_dataset=DATA_PATH,
        split="train[:100]",
        pad_token_id=tokenizer.tokenizer.eos_token_id,
        tokenizer=run.Config(AutoTokenizer, pretrained_model_name=args.model),
    )

    recipe.trainer.strategy = run.Config(nl.FSDP2Strategy, data_parallel_size=4, tensor_parallel_size=2)
    recipe.trainer.plugins = None
    executor = local_executor_torchrun(nodes=recipe.trainer.num_nodes, devices=recipe.trainer.devices)
    run.run(recipe, executor=executor)
