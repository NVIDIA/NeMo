import os
from datetime import datetime
from typing import Optional

import nemo_run as run
from utils import get_comm_overlap_callback_idx, hf_tokenizer, parse_cli_args, slurm_executor

from nemo.collections.llm.recipes.llama3_8b import pretrain_recipe
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
from nemo.lightning.run.plugins import NsysPlugin, PerfEnvPlugin


def llama3_8b_performance(
    compute_dtype: str,
    num_nodes: int,
    num_gpus_per_node: int,
    mbs: int,
    gbs: int,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    vp_size: Optional[int],
    max_steps: int,
):
    recipe = pretrain_recipe(performance_mode=True)

    # data module configs
    recipe.data.micro_batch_size = mbs
    recipe.data.global_batch_size = gbs
    recipe.data.num_train_samples = max_steps * (num_nodes * num_gpus_per_node)  # ensure only 1 epoch for whole run
    recipe.data.tokenizer = hf_tokenizer("meta-llama/Meta-Llama-3-8B")

    recipe.trainer.max_steps = max_steps
    recipe.trainer.num_nodes = num_nodes
    recipe.trainer.devices = num_gpus_per_node

    # parallelism configs
    recipe.trainer.strategy.tensor_model_parallel_size = tp_size
    recipe.trainer.strategy.pipeline_model_parallel_size = pp_size
    recipe.trainer.strategy.context_parallel_size = cp_size
    recipe.trainer.strategy.virtual_pipeline_model_parallel_size = vp_size
    if tp_size > 1:
        recipe.trainer.strategy.sequence_parallel = True
    else:
        recipe.trainer.strategy.sequence_parallel = False

    comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)

    # compute dtype configs
    if compute_dtype.lower() == "fp8":
        recipe.trainer.plugins = bf16_with_fp8_mixed()
    recipe.trainer.plugins.grad_reduce_in_fp32 = False  # bf16 grad dtype

    # callback configs
    garbage_collection_callback = run.Config(
        GarbageCollectionCallback,
        gc_interval_train=100,
        gc_interval_val=500,
    )
    recipe.trainer.callbacks.extend(
        [
            garbage_collection_callback,
        ]
    )
    dp_size = (num_nodes * num_gpus_per_node) / (tp_size * pp_size * cp_size)
    if dp_size > 1 and pp_size > 1 and vp_size and vp_size > 1:
        if comm_overlap_callback_idx >= 0:
            recipe.trainer.callbacks[comm_overlap_callback_idx].overlap_param_gather_with_optimizer_step = True

    return recipe


if __name__ == "__main__":
    args = parse_cli_args().parse_args()

    num_nodes = 1
    num_gpus_per_node = 8
    mbs = 1
    gbs = 128
    tp_size = 1
    pp_size = 1
    cp_size = 2
    vp_size = None
    max_steps = 100

    exp_name = f"llama3_8b_{args.compute_dtype}_{num_nodes}nodes_tp{tp_size}_pp{pp_size}_cp{cp_size}_vp{vp_size}_{mbs}mbs_{gbs}gbs"

    executor = slurm_executor(
        args.account,
        args.partition,
        args.log_dir,
        num_nodes,
        num_gpus_per_node,
        args.time_limit,
        args.container_image,
        custom_mounts=[],
        custom_env_vars={},
        retries=0,
    )

    recipe = llama3_8b_performance(
        args.compute_dtype,
        num_nodes,
        num_gpus_per_node,
        mbs,
        gbs,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        max_steps,
    )

    with run.Experiment(exp_name) as exp:
        exp.add(
            recipe,
            executor=executor,
            name=exp_name,
            plugins=[
                PerfEnvPlugin(enable_vboost=True),
                NsysPlugin(start_step=5, end_step=6),
            ],
        )

        if not args.dryrun:
            exp.run(sequential=True, detach=True)
        else:
            exp.dryrun()
