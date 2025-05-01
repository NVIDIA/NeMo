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

# NOTE: This script is only an example of using NeMo with NeMo-Run's APIs and is subject to change without notice.
# This script is used for pretraining on local and slurm executors.
# It uses NeMo 2.0 recipes (https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/llm/recipes/) and
# NeMo-Run (https://github.com/NVIDIA/NeMo-Run) to configure and execute the runs.

"""
Fine-tuning script for chat datasets. Uses the HuggingFace tokenizer chat template by default.

To finetune from a HuggingFace checkpoint, use nemo.collections.llm.import_ckpt to convert the checkpoint to a NeMo checkpoint.
When converting the checkpoint, the NeMo checkpoint will be saved in NEMO_HOME (set to ~/.cache/nemo by default).  If doing multi-node training, 
use `import_ckpt` with `output_path` set to a persistent directory, then use the same directory for `resume_path`.

If using a custom tokenizer, provide the HF tokenizer name in `--hf-tokenizer` and optionally a chat template in `--chat-template`, which will
override the default chat template. For example, this is useful when finetuning a base model to following instructions using the instruct model's tokenizer.

Note: to get the correct assistant mask from HF, you may need to customize the chat template to include a 'generation' keyword. See https://github.com/huggingface/transformers/pull/30650
"""
import argparse
from functools import partial
from typing import Any, Optional

import nemo_run as run

from nemo.collections import llm

# from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule
from nemo.collections.llm.gpt.data.chat import ChatDataModule
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer


def get_parser():
    parser = argparse.ArgumentParser(description="NeMo2.0 Pretraining")
    parser.add_argument(
        "--recipe",
        type=str,
        default="llama3_8b",
        help="Choose NeMo 2.0 recipe. Recipes are named in the format of <model_name>_<model_size>(_<long_sequence_length> or other special settings)",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        help="Path to the checkpoint to resume training from. If not provided, the model will be initialized from the HF model.",
        required=False,
    )
    # TODO add resume_dir?
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the finetuning chat dataset. Can be either ShareGPT or HuggingFace/OpenAI chat format",
        required=True,
    )
    parser.add_argument(
        "--ckpt-save-dir",
        type=str,
        help="Path to the directory to save the checkpoint.",
        required=True,
    )
    parser.add_argument(
        "--hf-tokenizer",
        type=str,
        help="Name of HF model to use for tokenizer.",
        required=False,
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        help="Path to the custom chat template to replace the HF tokenizer default chat template.",
        required=False,
    )
    parser.add_argument(
        "--peft-scheme",
        type=str,
        help="Name of the peft scheme to use for fine-tuning. Allowed values: 'lora'/'dora'/'none'/None.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        help="Global batch size.",
        required=False,
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        help="Micro batch size.",
        required=False,
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        help="Sequence length.",
        required=False,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Max steps.",
        required=False,
    )
    parser.add_argument(
        "--val-check-interval",
        type=int,
        help="Validation check interval.",
        required=False,
        default=100,
    )
    parser.add_argument(
        "--save-top-k",
        type=int,
        help="Save top k.",
        required=False,
        default=1,
    )
    parser.add_argument(
        "--tag",
        type=str,
        help="Optional tag for your experiment title which will be appended after the model/exp name.",
        required=False,
        default="",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dryrun and exit",
        default=False,
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Run on slurm using run.SlurmExecutor",
        default=False,
    )
    parser.add_argument(
        "--nodes",
        type=int,
        help="Number of nodes.",
        required=False,
    )
    parser.add_argument(
        "--devices",
        type=int,
        help="Number of devices.",
        required=False,
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        help="Tensor parallelism size.",
        required=False,
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        help="Pipeline model parallel size.",
        required=False,
    )
    parser.add_argument(
        "--expert-model-parallel-size",
        type=int,
        help="Expert model parallel size.",
        required=False,
    )
    parser.add_argument(
        "--expert-tensor-parallel-size",
        type=int,
        help="Expert tensor parallel size.",
        required=False,
    )
    return parser


def slurm_executor(
    user: str,
    host: str,
    remote_job_dir: str,
    account: str,
    partition: str,
    nodes: int,
    devices: int,
    time: str = "04:00:00",
    custom_mounts: Optional[list[str]] = None,
    custom_env_vars: Optional[dict[str, str]] = None,
    container_image: str = "nvcr.io/nvidia/nemo:dev",
    retries: int = 0,
    dependency=None,
) -> run.SlurmExecutor:
    if not (user and host and remote_job_dir and account and partition and nodes and devices):
        raise RuntimeError(
            "Please set user, host, remote_job_dir, account, partition, nodes and devices args for using this ",
            "function.",
        )

    mounts = []
    if custom_mounts:
        mounts.extend(custom_mounts)

    env_vars = {
        "TRANSFORMERS_OFFLINE": "1",  # Enable online downloads from HuggingFace
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",  # Disable caching NCCL communication buffer memory
        "NCCL_NVLS_ENABLE": "0",  # Disable NVLink SHARP to save memory
    }
    if custom_env_vars:
        env_vars |= custom_env_vars

    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.SSHTunnel(
            user=user,
            host=host,
            job_dir=remote_job_dir,
        ),
        nodes=nodes,
        ntasks_per_node=devices,
        gpus_per_node=devices,
        mem="0",
        exclusive=True,
        gres="gpu:8",
        packager=run.GitArchivePackager(),
    )

    executor.container_image = container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.retries = retries
    executor.time = time

    return executor


def local_executor_torchrun(nodes: int = 1, devices: int = 2) -> run.LocalExecutor:
    env_vars = {
        "TRANSFORMERS_OFFLINE": "1",  # Enable online downloads from HuggingFace
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",  # Disable caching NCCL communication buffer memory
        "NCCL_NVLS_ENABLE": "0",  # Disable NVLink SHARP to save memory
        "NVTE_FUSED_ATTN": "1",  # Disable cuDNN fused attention
    }

    executor = run.LocalExecutor(ntasks_per_node=devices, launcher="torchrun", env_vars=env_vars)

    return executor


def main():
    args = get_parser().parse_args()
    if args.tag and not args.tag.startswith("-"):
        args.tag = "-" + args.tag

    exp_name = args.recipe

    # Uses configs from NeMo directly
    assert hasattr(
        llm, args.recipe
    ), f"Recipe named {args.recipe} not found. General format is <model_name>_<model_size>(_<long_sequence_length> or other special settings)"
    finetune_recipe = getattr(llm, args.recipe).finetune_recipe
    finetune = partial(finetune_recipe)(name=exp_name, dir=args.ckpt_save_dir, peft_scheme=args.peft_scheme)
    finetune.tokenizer = 'data'  # by default use dataset's HF tokenizer
    if not args.hf_tokenizer:
        # Use base model tokenizer if not specified
        args.hf_tokenizer = finetune.resume.restore_config.path.removeprefix("nemo://")
    if args.resume_path:
        finetune.resume.restore_config.path = args.resume_path

    if args.chat_template and 'generation' not in args.chat_template:
        raise ValueError(
            "Please ensure the chat template includes a 'generation' keyword for proper assistant mask during training. See https://github.com/huggingface/transformers/pull/30650"
        )
    tokenizer = run.Config(
        get_nmt_tokenizer, library='huggingface', model_name=args.hf_tokenizer, chat_template=args.chat_template
    )

    finetune.trainer.val_check_interval = args.val_check_interval
    finetune.log.ckpt.save_top_k = args.save_top_k
    if args.max_steps:
        finetune.trainer.max_steps = args.max_steps
    if args.nodes:
        finetune.trainer.num_nodes = args.nodes
    if args.devices:
        finetune.trainer.devices = args.devices
    if args.tensor_parallel_size:
        finetune.trainer.strategy.tensor_model_parallel_size = args.tensor_parallel_size
    if args.pipeline_parallel_size:
        finetune.trainer.strategy.pipeline_model_parallel_size = args.pipeline_parallel_size
    if args.expert_model_parallel_size:
        finetune.trainer.strategy.expert_model_parallel_size = args.expert_model_parallel_size
    if args.expert_tensor_parallel_size:
        finetune.trainer.strategy.expert_tensor_parallel_size = args.expert_tensor_parallel_size

    # Change here and add your files to custom_mounts
    finetune.data = run.Config(
        ChatDataModule,
        dataset_root=args.data_path,
        seq_length=args.seq_length if args.seq_length else finetune.data.seq_length,
        tokenizer=tokenizer,
        global_batch_size=args.global_batch_size if args.global_batch_size else finetune.data.global_batch_size,
        micro_batch_size=args.micro_batch_size if args.micro_batch_size else finetune.data.micro_batch_size,
        use_hf_tokenizer_chat_template=True,
    )

    executor: run.Executor

    if args.slurm:
        # TODO: Set your custom parameters for the Slurm Executor.
        executor = slurm_executor(
            user="jennifchen",
            host="cw-dfw-cs-001-login-01",
            remote_job_dir="/lustre/fsw/portfolios/coreai/users/jennifchen",
            account="coreai_dlalgo_modelopt",
            partition="batch",
            nodes=finetune.trainer.num_nodes,
            devices=finetune.trainer.devices,
            custom_mounts=[
                "/lustre/fsw:/lustre/fsw",
                "/lustre/fsw/portfolios/coreai/users/jennifchen/code/NeMo:/opt/NeMo",
                "/lustre/fsw/portfolios/coreai/users/jennifchen/code/Megatron-LM:/opt/megatron-lm",
                "/lustre/fsw/portfolios/coreai/users/jennifchen/code/modelopt/modelopt:/usr/local/lib/python3.12/dist-packages/modelopt",
            ],
            custom_env_vars={"HF_HOME": "/lustre/fsw/portfolios/coreai/users/jennifchen/hf_cache"},
        )
    else:
        executor = local_executor_torchrun(nodes=finetune.trainer.num_nodes, devices=finetune.trainer.devices)

    with run.Experiment(f"{exp_name}{args.tag}") as exp:
        for i in range(1):
            exp.add(
                finetune,
                executor=executor,
                name=exp_name,
                tail_logs=True if isinstance(executor, run.LocalExecutor) else False,
            )

        if args.dryrun:
            exp.dryrun()
        else:
            exp.run(sequential=True, detach=True)


if __name__ == "__main__":
    main()
