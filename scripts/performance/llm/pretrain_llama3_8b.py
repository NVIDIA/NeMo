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

import nemo_run as run

from nemo.collections.llm.recipes.llama3_8b import pretrain_recipe
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

from ..helpers import (
    set_exp_logging_configs,
    set_primary_perf_configs,
)
from ..utils import hf_tokenizer


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
    use_mcore_fsdp: bool,
    recompute_layers: int,
    activation_offload_layers: int,
    recompute_modules: list,
    keep_fsdp_fp8_transpose_cache: bool,
    use_user_buffer_registration: bool,
    use_sharp: bool,
):
    """
    llama3 8b pre-train recipe aimed at achieving best possible performance and faster
    overall runtime.

    NOTE: Use fp8 precision training with caution. It might not give desirable results.
    """
    recipe = pretrain_recipe(performance_mode=True)
    recipe = set_primary_perf_configs(
        recipe,
        "pre_train",
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
        nccl_communicator_config_path=args.nccl_communicator_config_path,
        use_mcore_fsdp=use_mcore_fsdp,
        use_fsdp_double_buffer=args.use_fsdp_double_buffer,
        use_user_buffer_registration=use_user_buffer_registration,
        use_sharp=use_sharp,
        keep_fsdp_fp8_transpose_cache=keep_fsdp_fp8_transpose_cache,
    )
    recipe = set_exp_logging_configs(
        recipe, "pre_train", "llm", "llama3", args.tensorboard, args.wandb, args.wandb_prj_name, args.wandb_job_name
    )

    # data module configs
    if args.use_hf_tokenizer:
        recipe.data.tokenizer = hf_tokenizer("meta-llama/Meta-Llama-3-8B")
    else:
        recipe.data.tokenizer = run.Config(
            get_nmt_tokenizer, library="null", model_name="NullTokenizer", vocab_size=128256
        )
        recipe.model.tokenizer = recipe.data.tokenizer

    return recipe


if __name__ == "__main__":
    from .main import run_performance_experiment
    run_performance_experiment(
        task="pre_train",
        model="llama3",
        model_size="8b",
        override_recipe_configs=override_recipe_configs,
        base_recipe=pretrain_recipe(performance_mode=False),
        default_perf_recipe=pretrain_recipe(performance_mode=True),
    )