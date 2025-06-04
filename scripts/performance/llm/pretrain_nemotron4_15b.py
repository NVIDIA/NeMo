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

import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
import nemo_run as run

from nemo.collections.llm.recipes.nemotron4_15b import pretrain_recipe
from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import userbuffers_bf16_b200_h6144_tp2_mbs1_seqlen4096
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.utils import logging

from ..utils import (
    get_comm_overlap_callback_idx,
    run_performance_experiment,
    set_exp_logging_configs,
    set_primary_perf_configs,
)


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
    **kwargs,
):
    """
    nemotron4 15b pre-train recipe aimed at achieving best possible performance.

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
        enable_cuda_graphs=enable_cuda_graphs,
        use_user_buffer_registration=args.use_user_buffer_registration,
        use_sharp=args.use_sharp,
        compute_dtype=args.compute_dtype,
        fp8_recipe=args.fp8_recipe,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
    )
    recipe = set_exp_logging_configs(
        recipe, "pre_train", "llm", "nemotron", args.tensorboard, args.wandb, args.wandb_prj_name, args.wandb_job_name
    )

    gpu_type = args.gpu.lower()

    # data module configs
    if args.use_hf_tokenizer:
        logging.warning("HuggingFace tokenizer not supported for Nemotron4 15B. Using NullTokenizer.")
    recipe.data.tokenizer = run.Config(
        get_nmt_tokenizer, library="null", model_name="NullTokenizer", vocab_size=256000
    )
    recipe.model.tokenizer = recipe.data.tokenizer

    comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
    assert comm_overlap_callback_idx is not None, "MegatronCommOverlapCallback missing. Required for performance."

    if gpu_type in ["b200", "gb200"]:
        if args.fp8_recipe.lower() != "mxfp8":
            tp_comm_overlap_cfg = userbuffers_bf16_b200_h6144_tp2_mbs1_seqlen4096
            # needed as tp_overlap_configs.userbuffers are dataclass objects which are unserializable
            tp_comm_overlap_cfg = fdl.cast(run.Config, fdl_dc.convert_dataclasses_to_configs(tp_comm_overlap_cfg))
            recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap_cfg = tp_comm_overlap_cfg
    if args.compute_dtype.lower() == "fp8" and args.fp8_recipe.lower() == "mxfp8":
        recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap_cfg = None
        recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap = False

    return recipe


if __name__ == "__main__":
    run_performance_experiment(
        "pre_train",
        "nemotron4",
        "15b",
        override_recipe_configs,
    )
