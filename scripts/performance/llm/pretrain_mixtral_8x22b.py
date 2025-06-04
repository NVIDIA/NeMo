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
from typing import Optional

import nemo_run as run

from nemo.collections.llm.recipes.mixtral_8x22b_64k import pretrain_recipe
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.utils import logging

from ..utils import hf_tokenizer, run_performance_experiment, set_exp_logging_configs, set_primary_perf_configs


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
    etp_size: Optional[int],
    enable_cuda_graphs: bool,
    use_mcore_fsdp: bool,
    recompute_layers: int,
    activation_offload_layers: int,
    **kwargs,
):
    """
    mixtral 8x22b pre-train recipe aimed at achieving best possible performance.

    NOTE: Use fp8 precision training with caution. It might not give desirable results.
    """
    # print out ignored kwarg warnings
    for k, v in kwargs.items():
        logging.warning(f"{splitext(basename(__file__))[0]}.override_recipe_configs: {k} = {v} is ignored")

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
        etp_size,
        enable_cuda_graphs,
        use_mcore_fsdp,
        use_user_buffer_registration=args.use_user_buffer_registration,
        use_sharp=args.use_sharp,
        recompute_layers=recompute_layers,
        activation_offload_layers=activation_offload_layers,
        compute_dtype=args.compute_dtype,
        fp8_recipe=args.fp8_recipe,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
    )
    recipe = set_exp_logging_configs(
        recipe, "pre_train", "llm", "mixtral", args.tensorboard, args.wandb, args.wandb_prj_name, args.wandb_job_name
    )

    # data module configs
    if args.use_hf_tokenizer:
        recipe.data.tokenizer = hf_tokenizer("mistralai/Mixtral-8x22B-v0.1")
    else:
        recipe.data.tokenizer = run.Config(
            get_nmt_tokenizer, library="null", model_name="NullTokenizer", vocab_size=32000
        )
        recipe.model.tokenizer = recipe.data.tokenizer

    # to mitigate the incorrect gradient_scaling_factor calculation in megatron.core
    # under scenario average_in_collective=True and tp_size != etp_size, disabling average_in_collective.
    if etp_size is not None and etp_size != tp_size:
        recipe.trainer.strategy.ddp.average_in_collective = False

    return recipe


if __name__ == "__main__":
    run_performance_experiment(
        "pre_train",
        "mixtral",
        "8x22b",
        override_recipe_configs,
    )
