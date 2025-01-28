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
from argument_parser import parse_cli_args
from utils import (
    get_comm_overlap_callback_idx,
    get_performance_configs,
    hf_tokenizer,
    set_recipe_primary_configs,
    slurm_executor,
)

from nemo.collections.llm.recipes.llama3_8b import pretrain_recipe
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
from nemo.lightning.run.plugins import NsysPlugin, PerfEnvPlugin


def llama3_8b_performance_recipe(
    compute_dtype: str,
    num_nodes: int,
    num_gpus_per_node: int,
    max_steps: int,
    mbs: int,
    gbs: int,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    vp_size: int,
    ep_size: int,
):
    """
    llama3 8b pre-train recipe aimed at achieving best possible performance and faster
    overall runtime.

    NOTE: Use fp8 precision training with caution. It might not give desirable results.
    """
    recipe = pretrain_recipe(performance_mode=True)
    recipe = set_recipe_primary_configs(
        recipe, num_nodes, num_gpus_per_node, mbs, gbs, max_steps, tp_size, pp_size, cp_size, vp_size, ep_size
    )

    # data module configs
    recipe.data.num_train_samples = max_steps * gbs * mbs  # ensure only 1 epoch for whole run
    recipe.data.tokenizer = hf_tokenizer("meta-llama/Meta-Llama-3-8B")

    comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)

    # compute dtype configs
    if compute_dtype.lower() == "fp8":
        recipe.trainer.plugins = bf16_with_fp8_mixed()

    # callback configs
    garbage_collection_callback = run.Config(
        GarbageCollectionCallback,
        gc_interval_train=100,
        gc_interval_val=100,
    )
    recipe.trainer.callbacks.extend(
        [
            garbage_collection_callback,
        ]
    )
    dp_size = (num_nodes * num_gpus_per_node) / (tp_size * pp_size * cp_size)
    if comm_overlap_callback_idx is not None:
        # WARNING: If True, checkpointing (if enabled) might not work
        recipe.trainer.callbacks[comm_overlap_callback_idx].overlap_param_gather_with_optimizer_step = bool(
            dp_size > 1 and pp_size > 1 and vp_size and vp_size > 1
        )

    # Misc. for overall faster experiment runtime
    recipe.log.ckpt = None
    recipe.trainer.enable_checkpointing = False
    recipe.trainer.val_check_interval = max_steps
    recipe.trainer.log_every_n_steps = 1

    return recipe


if __name__ == "__main__":
    args = parse_cli_args().parse_args()

    kwargs = get_performance_configs(args.gpu.lower(), "llama3", "8b", args)
    num_nodes, mbs, gbs, tp_size, pp_size, cp_size, vp_size, ep_size = kwargs

    exp_params = [
        f"{splitext(basename(__file__))}_{args.compute_dtype}",
        f"{num_nodes}nodes_tp{tp_size}_pp{pp_size}_cp{cp_size}_vp{vp_size}_{ep_size}_{mbs}mbs_{gbs}gbs",
    ]
    exp_name = "_".join(exp_params)

    executor = slurm_executor(
        args.account,
        args.partition,
        args.log_dir,
        num_nodes,
        args.devices_per_node,
        args.time_limit,
        args.container_image,
        custom_mounts=[],
        custom_env_vars={},
        hf_token=args.hf_token,
        nemo_home=args.nemo_home,
    )

    recipe = llama3_8b_performance_recipe(
        args.compute_dtype,
        num_nodes,
        args.devices_per_node,
        args.max_steps,
        mbs,
        gbs,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
    )

    if not args.tensorboard:  # tensorboard adds performance overhead.
        recipe.log.tensorboard = None
        recipe.trainer.logger = False
    else:
        # default path is NOT intuitive- `<log_dir>/code/nemo_experiments/tb_logs/default/<tfevents_file>`
        # following line ensures file is at- `<log_dir>/lightning_logs/tb_logs/default/<tfevents_file>`
        recipe.log.log_dir = "/nemo_run/lightning_logs"

    plugins = [
        PerfEnvPlugin(
            enable_vboost=True,
            nccl_pp_comm_chunksize=2097152 if pp_size > 1 else None,
        )
    ]
    if args.enable_profiling:
        plugins.append(NsysPlugin(start_step=5, end_step=6))

    with run.Experiment(splitext(basename(__file__))) as exp:
        exp.add(recipe, executor=executor, name=exp_name, plugins=plugins)

        if not args.dryrun:
            exp.run(sequential=True, detach=True)
        else:
            exp.dryrun()
