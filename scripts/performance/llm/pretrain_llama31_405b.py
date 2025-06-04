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

from nemo.collections.llm.recipes.llama31_405b import pretrain_recipe
from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import (
    userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192,
    userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
    userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192,
    userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.utils import logging

from ..utils import (
    get_comm_overlap_callback_idx,
    hf_tokenizer,
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
    use_mcore_fsdp: bool,
    recompute_layers: int,
    activation_offload_layers: int,
    **kwargs,
):
    """
    llama3 405b pre-train recipe aimed at achieving best possible performance.

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
        use_mcore_fsdp=use_mcore_fsdp,
        use_user_buffer_registration=args.use_user_buffer_registration,
        use_sharp=args.use_sharp,
        recompute_layers=recompute_layers,
        activation_offload_layers=activation_offload_layers,
        compute_dtype=args.compute_dtype,
        fp8_recipe=args.fp8_recipe,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
    )
    recipe = set_exp_logging_configs(
        recipe, "pre_train", "llm", "llama3", args.tensorboard, args.wandb, args.wandb_prj_name, args.wandb_job_name
    )

    gpu_type = args.gpu.lower()

    # data module configs
    if args.use_hf_tokenizer:
        recipe.data.tokenizer = hf_tokenizer("meta-llama/Llama-3.1-405B")
    else:
        recipe.data.tokenizer = run.Config(
            get_nmt_tokenizer, library="null", model_name="NullTokenizer", vocab_size=128256
        )
        recipe.model.tokenizer = recipe.data.tokenizer

    userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192.qkv_fprop.aggregate = False
    userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192.proj_dgrad.aggregate = False
    userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192.fc1_fprop.aggregate = False
    userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192.fc2_dgrad.aggregate = False

    userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192.qkv_fprop.aggregate = False
    userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192.proj_dgrad.aggregate = False
    userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192.fc1_fprop.aggregate = False
    userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192.fc2_dgrad.aggregate = False

    ub_cfg = {
        "h100": {
            "bf16": userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192,
        },
        "b200": {
            "bf16": userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192,
        },
        "gb200": {
            "bf16": userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192,
        },
    }

    comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
    assert comm_overlap_callback_idx is not None, "MegatronCommOverlapCallback missing. Required for performance."

    if args.fp8_recipe.lower() != "mxfp8":
        tp_comm_overlap_cfg = ub_cfg[gpu_type][args.compute_dtype]
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
        "llama31",
        "405b",
        override_recipe_configs,
        custom_env_vars={
            "NVTE_NORM_FWD_USE_CUDNN": "1",
            "NVTE_NORM_BWD_USE_CUDNN": "1",
        },  # for properly overlapping normalization kernels with FSDP communication
    )
