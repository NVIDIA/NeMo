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

from os.path import basename, splitext

import nemo_run as run

from nemo.collections.llm.recipes.llama3_8b import finetune_recipe, model

from ..helpers import (
    set_exp_logging_configs,
    set_primary_perf_configs,
)
from ..utils import hf_tokenizer
from ..main import run_performance_experiment

HF_MODEL_URI = "meta-llama/Meta-Llama-3-8B"

# Set this to True if checkpoint is available at 'NEMO_HOME'. If set to False,
# extra Slurm job will be scheduled. In this case, if checkpoint is available
# at 'NEMO_HOME', fine-tuning job will use this checkpoint, else, it will be
# downloaded from HuggingFace
SKIP_IMPORT = False

# Set this to True if dataset is already downloaded. If set to False,
# dataset will be downloaded from HuggingFace
SKIP_DATASET_DOWNLOAD = False


def override_recipe_configs(
    args: str,
    num_nodes: int,
    mbs: int,
    gbs: int,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    vp_size: int,
    ep_size: int,
    enable_cuda_graphs: bool,
):
    """
    llama3 8b pre-train recipe aimed at achieving best possible performance.

    NOTE: Use fp8 precision training with caution. It might not give desirable results.
    """
    finetuning_scheme = "none" if args.finetuning == "sft" else args.finetuning

    gpu_type = args.gpu.lower()
    if gpu_type in ["b200", "gb200"]:
        recipe = finetune_recipe(peft_scheme=finetuning_scheme, performance_mode=True, seq_length=16384)
    else:
        recipe = finetune_recipe(peft_scheme=finetuning_scheme, performance_mode=True)

    recipe = set_primary_perf_configs(
        recipe,
        finetuning_scheme,
        num_nodes,
        args.gpus_per_node,
        mbs,
        gbs,
        args.max_steps,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
        enable_cuda_graphs=enable_cuda_graphs,
        compute_dtype=args.compute_dtype,
        fp8_recipe=args.fp8_recipe,
        use_mcore_fsdp=args.use_mcore_fsdp,
        use_fsdp_double_buffer=args.use_fsdp_double_buffer,
        use_user_buffer_registration=args.use_user_buffer_registration,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
        use_sharp=args.use_sharp,
    )
    recipe = set_exp_logging_configs(
        recipe,
        finetuning_scheme,
        "llm",
        "llama3",
        args.tensorboard,
        args.wandb,
        args.wandb_prj_name,
        args.wandb_job_name,
    )

    # data module configs
    recipe.data.tokenizer = hf_tokenizer(HF_MODEL_URI)

    recipe.optim.config.use_distributed_optimizer = True
    recipe.model.config.disable_parameter_transpose_cache = True

    return recipe


if __name__ == "__main__":
    run_performance_experiment(
        task="finetune",
        model="llama3",
        model_size="8b",
        override_recipe_configs=override_recipe_configs,
        finetuning_skip_import=SKIP_IMPORT,
        hf_model_url=HF_MODEL_URI,
        base_recipe=finetune_recipe(performance_mode=False),
        default_perf_recipe=finetune_recipe(performance_mode=True),
    )