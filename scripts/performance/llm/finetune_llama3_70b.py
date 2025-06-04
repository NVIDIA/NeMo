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

from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.recipes.llama3_70b import finetune_recipe
from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import (
    userbuffers_fp8_h100_h8192_tp2_mbs1_seqlen4096_lora,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.utils import logging

from ..argument_parser import parse_cli_args
from ..utils import (
    get_comm_overlap_callback_idx,
    hf_tokenizer,
    isfile_train_pack_metadata,
    run_performance_experiment,
    set_exp_logging_configs,
    set_primary_perf_configs,
)

HF_MODEL_URI = "meta-llama/Meta-Llama-3-70B"

# Set this to True if checkpoint is available at 'NEMO_HOME'. If set to False,
# extra Slurm job will be scheduled. In this case, if checkpoint is available
# at 'NEMO_HOME', fine-tuning job will use this checkpoint, else, it will be
# downloaded from HuggingFace
SKIP_IMPORT = False


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
    llama3 70b fine-tuning recipe aimed at achieving best possible performance.

    NOTE: Use fp8 precision training with caution. It might not give desirable results.
    """
    # print out ignored kwarg warnings
    for k, v in kwargs.items():
        logging.warning(f"{splitext(basename(__file__))[0]}.override_recipe_configs: {k} = {v} is ignored")

    finetuning_scheme = "none" if args.finetuning == "sft" else args.finetuning
    gpu_type = args.gpu.lower()
    if gpu_type in ["gb200"] and finetuning_scheme == "lora":
        # On GB200 for lora task, we need to enable Cuda Graph for optimal performance.
        # However, Cuda Graph increases memory usage, so in order to avoid OOM, we need
        # to reduce the sequence length.
        recipe = finetune_recipe(peft_scheme=finetuning_scheme, performance_mode=True, seq_length=2048)
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
        use_mcore_fsdp=use_mcore_fsdp,
        recompute_layers=recompute_layers,
        activation_offload_layers=activation_offload_layers,
        compute_dtype=args.compute_dtype,
        fp8_recipe=args.fp8_recipe,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
        use_user_buffer_registration=args.use_user_buffer_registration,
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
    if args.use_hf_tokenizer:
        recipe.data.tokenizer = hf_tokenizer(HF_MODEL_URI)
    else:
        recipe.data.tokenizer = run.Config(
            get_nmt_tokenizer, library="null", model_name="NullTokenizer", vocab_size=128256
        )
        recipe.model.tokenizer = recipe.data.tokenizer
    if recipe.data.__fn_or_cls__ == SquadDataModule and not isfile_train_pack_metadata(HF_MODEL_URI, recipe.data):
        # flag is valid only for SquadDataModule
        recipe.data.force_redownload = True

    comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
    assert comm_overlap_callback_idx is not None, "MegatronCommOverlapCallback missing. Required for performance."

    if (
        finetuning_scheme == "lora"
        and tp_size > 1
        and args.compute_dtype.lower() == "fp8"
        and args.fp8_recipe.lower() != "mxfp8"
    ):
        tp_comm_overlap_cfg = userbuffers_fp8_h100_h8192_tp2_mbs1_seqlen4096_lora if tp_size == 2 else None
        if tp_comm_overlap_cfg:
            # Enable TP comm overlap with the given config
            recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap = True
            tp_comm_overlap_cfg = fdl.cast(run.Config, fdl_dc.convert_dataclasses_to_configs(tp_comm_overlap_cfg))
            recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap_cfg = tp_comm_overlap_cfg

            # Disable this overlap to allow skipping an all-gather which is redundant for LoRA
            recipe.model.config.tp_comm_overlap_disable_qkv = True

            # Allow overlapping of dgrad reduce-scatter with dgrad GEMMs
            # (instead of wgrad GEMMs which are not done when using LoRA)
            recipe.model.config.tp_comm_bulk_dgrad = False
            recipe.model.config.tp_comm_overlap_rs_dgrad = True
    if args.compute_dtype.lower() == "fp8" and args.fp8_recipe.lower() == "mxfp8":
        recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap_cfg = None
        recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap = False

    recipe.optim.config.use_distributed_optimizer = True
    recipe.model.config.disable_parameter_transpose_cache = True

    return recipe


if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    run_performance_experiment(
        args.finetuning,
        "llama3",
        "70b",
        finetune_skip_import=SKIP_IMPORT,
        hf_model_uri=HF_MODEL_URI,
    )
