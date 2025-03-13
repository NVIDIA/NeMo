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

# NOTE: This script is just an example of using NeMo checkpoints
# for generating outputs and is subject to change without notice.

from argparse import ArgumentParser

import torch
import torch.distributed
from megatron.core.inference.common_inference_params import CommonInferenceParams

import nemo.lightning as nl
from nemo.collections.llm import api

"""
torchrun --nproc-per-node=8 /lustre/fsw/portfolios/coreai/users/ataghibakhsh/NeMo/scripts/llm/generate.py \
    --model_path=/lustre/fsw/portfolios/coreai/users/ataghibakhsh/final_nm5/nm5_56b_base_8k \
    --tp=8 \
    --devices=8 \
    --num_tokens_to_generate=40 \
    --temperature=0.01 \
    --fp8

torchrun --nproc-per-node=1 /lustre/fsw/portfolios/coreai/users/ataghibakhsh/NeMo/scripts/llm/generate.py \
    --model_path=/lustre/fsw/portfolios/coreai/users/ataghibakhsh/final_nm5/final_algined_8b_nemo2 \
    --tp=1 \
    --devices=1 \
    --num_tokens_to_generate=40 \
    --temperature=0.01 \
    --fp8
"""

def get_args():
    """
    Parse the command line arguments.
    """
    parser = ArgumentParser(description="""Run generation on a few sample prompts given the checkpoint path.""")
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["Q: How are you?",
                 "Q: How big is the universe?",
                 "Q: How is the weather?",
                 "Q: How many stars are there?",
                 "Paris is know for its ",
                 "In a hot sunny day, you should ",
                 "the biggest ocean on Earth is called ",
                 "Q: How old are you?",
                 ],
        help="List of prompt strings",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="""Path to NeMo 2 checkpoint""",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="""Tensor parallel size""",
    )
    parser.add_argument(
        "--pp",
        type=int,
        default=1,
        help="""Pipeline parallel size""",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="""Number of GPUs to use on a single node""",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="""Number of nodes to use""",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="""Temperature to be used in megatron.core.inference.common_inference_params.CommonInferenceParams""",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="""top_p to be used in megatron.core.inference.common_inference_params.CommonInferenceParams""",
    )
    parser.add_argument(
        "--top_k",
        type=float,
        default=0,
        help="""top_k to be used in megatron.core.inference.common_inference_params.CommonInferenceParams""",
    )
    parser.add_argument(
        "--add_BOS",
        action="store_true",
        help="""Whether to add BOS token to the prompt""",
    )
    parser.add_argument(
        "--num_tokens_to_generate",
        type=int,
        default=25,
        help="""Number of tokens to generate per prompt""",
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="""Whether to run inference in FP8 precision""",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
        help="""Maximum batch size for inference""",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=1234,
        help="""Random seed for generation""",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    if args.fp8:
        assert len(args.prompts) % 8 == 0, "Batch size should be divisible by 8 for FP8 inference"
    
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp,
        pipeline_model_parallel_size=args.pp,
        context_parallel_size=1,
        sequence_parallel=False,
        setup_optimizers=False,
        store_optimizer_states=False,
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.nodes,
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
            fp8="hybrid" if args.fp8 else None,
            fp8_amax_history_len=1,
            fp8_amax_compute_algo="max",
        ),
    )
    prompts=args.prompts
    ############################
    from nemo.collections.llm.inference import chat_utils

    prompts=[{"role": "system", "content": ""}, 
            {"role": "user","content":"Write a limerick about the wonders of GPU computing."},
            {"role": "assistant", "content": "There once was a chip full of might, \
              That made data dance day and night. \
              With GPUs so grand, \
              Tasks were done on demand, \
              Turning code into speed and delight!"},
            {"role": "user", "content": "Can you change it to use the word NVIDIA?"}]
    
    prompts = prompts + [
            {'role': 'assistant', 'content': ''}
        ]  # adding trailing assistant message so that prompt ends with Assistant tag.
    special_tokens = chat_utils.NM5_CHAT_PROMPT_TOKENS
    nemo_source = chat_utils.convert_messages(prompts)
    header, conversation, data_type, mask_role = chat_utils.get_header_conversation_type_mask_role(
        nemo_source, special_tokens
    )
    len_strip = len(special_tokens['end_of_turn'] + special_tokens['turn_start'])
    prompts = [conversation[:-len_strip]]*8

    ############################
    results = api.generate(
        path=args.model_path,
        prompts=prompts,
        trainer=trainer,
        add_BOS=args.add_BOS,
        inference_params=CommonInferenceParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_tokens_to_generate=args.num_tokens_to_generate,
        ),
        text_only=True,
        max_batch_size=args.max_batch_size,
        random_seed=args.random_seed,
    )
    if torch.distributed.get_rank() == 0:
        for i, r in enumerate(results):
            print(prompts[i])
            print("*" * 50)
            print(r)
            print("\n\n")
