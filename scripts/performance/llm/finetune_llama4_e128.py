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
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed
from nemo.collections.llm.gpt.data.squad import SquadDataModule
from nemo.collections.llm.recipes.llama4_e128 import finetune_recipe, model
from nemo.lightning.run.plugins import NsysPlugin, PerfEnvPlugin

from ..argument_parser import parse_cli_args
from ..utils import (
    args_sanity_check,
    get_comm_overlap_callback_idx,
    get_user_configs,
    hf_tokenizer,
    import_ckpt_experiment,
    isfile_train_pack_metadata,
    set_exp_logging_configs,
    set_primary_perf_configs,
    slurm_executor,
    prepare_squad_dataset_experiment,
)

HF_MODEL_URI = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"


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
    num_layers: int,
    hidden_size: int,
    etp_size: int,
    enable_cuda_graphs: bool,
    use_mcore_fsdp: bool,
    recompute_layers: int,
    activation_offload_layers: int,
):
    """
    Llama4 e128 fine-tuning recipe aimed at achieving best possible performance.
    Only supports SFT (Supervised Fine-Tuning).

    NOTE: Use fp8 precision training with caution. It might not give desirable results.
    """
    finetuning_scheme = "none" if args.finetuning == "sft" else args.finetuning
    #assert finetuning_scheme != "lora"

    recipe = finetune_recipe(peft_scheme=finetuning_scheme, performance_mode=True, packed_sequence=True)

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
        num_layers,
        hidden_size,
        etp_size,
        enable_cuda_graphs=enable_cuda_graphs,
        use_mcore_fsdp=use_mcore_fsdp,
        recompute_layers=recompute_layers,
        activation_offload_layers=activation_offload_layers,
        compute_dtype=args.compute_dtype,
        fp8_recipe=args.fp8_recipe,
    )

    recipe = set_exp_logging_configs(
        recipe, finetuning_scheme, "llm", "llama4", args.tensorboard, args.wandb, args.wandb_prj_name, args.wandb_job_name
    )


    # data module configs
    if args.use_hf_tokenizer:
        recipe.data.tokenizer = hf_tokenizer(HF_MODEL_URI)

    # Compute dtype configs
    if args.compute_dtype.lower() == "fp8":
        recipe.trainer.plugins = bf16_with_fp8_mixed()
        recipe.trainer.plugins.grad_reduce_in_fp32 = False

    recipe.model.config.cross_entropy_fusion_impl = "te"
    recipe.model.config.cross_entropy_loss_fusion = True
    recipe.model.config.apply_rope_fusion = True
    recipe.model.config.moe_permute_fusion = True
    return recipe

if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    args_sanity_check(args)

    kwargs = get_user_configs(args.gpu.lower(), "sft", "llama4", "e128", args)
    (
        num_nodes,
        mbs,
        gbs,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
        num_layers,
        hidden_size,
        etp_size,
        enable_cuda_graphs,
        use_mcore_fsdp,
        recompute_layers,
        activation_offload_layers,
    ) = kwargs[0:15]

    recipe = override_recipe_configs(
        args,
        num_nodes,
        mbs,
        gbs,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
        num_layers,
        hidden_size,
        etp_size,
        enable_cuda_graphs,
        use_mcore_fsdp,
        recompute_layers,
        activation_offload_layers,
    )
    exp_config = (
        f"{num_nodes}nodes_tp{tp_size}_pp{pp_size}_cp{cp_size}_vp{vp_size}_ep{ep_size}_etp{etp_size}_{mbs}mbs_{gbs}gbs"
    )
    exp_name = f"{splitext(basename(__file__))[0]}_{args.compute_dtype}_{exp_config}"

    plugins = [
        PerfEnvPlugin(
            enable_vboost=True,
            nccl_pp_comm_chunksize=2097152 if pp_size > 1 else None,
            gpu_sm100_or_newer=(args.gpu.lower() in ['b200', 'gb200']),
        )
    ]

    executor = slurm_executor(
        args.account,
        args.partition,
        args.log_dir,
        num_nodes,
        args.gpus_per_node,
        args.time_limit,
        args.container_image,
        custom_mounts=args.custom_mounts,
        custom_env_vars={},
        hf_token=args.hf_token,
        nemo_home=args.nemo_home,
        wandb_key=args.wandb_key,
    )

    with run.Experiment(exp_name) as exp:
        if not args.skip_import_checkpoint:
            assert args.hf_token is not None, "HF token is required for importing checkpoint from HuggingFace"
            exp.add(*import_ckpt_experiment(executor, model(), source=f"hf://{HF_MODEL_URI}"))
        if not args.skip_dataset_download:    
            exp.add(*prepare_squad_dataset_experiment(executor, HF_MODEL_URI, seq_length=4096))
        if not args.skip_finetuning:
            exp.add(
                recipe,
                executor=executor,
                name=exp_name,
                plugins=plugins,
            )
            if not args.dryrun:
                exp.run(sequential=True, detach=True)
            else:
                exp.dryrun()