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

from nemo.lightning.run.plugins import MemoryProfilePlugin, NsysPlugin, PerfEnvPlugin

from .argument_parser import parse_cli_args
from .executors import local_executor, slurm_executor
from .helpers import build_perf_env_plugin, get_user_configs


def run_performance_experiment(task: str, model: str, model_size: str, override_recipe_configs: Callable, **kwargs):
    """Run performance experiment. It launches nemo experiment."""

    # parse arguments
    args = parse_cli_args().parse_args()
    args_sanity_check(args)

    # get user configs
    config_args = get_user_configs(args.gpu.lower(), task, model, model_size, args)

    # override recipe configs
    recipe = override_recipe_configs(args, *config_args)

    # set experiment name
    (
        num_nodes,
        mbs,
        gbs,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
        etp_size,
    ) = config_args[:9]
    exp_config = f"{num_nodes}nodes_tp{tp_size}_pp{pp_size}_cp{cp_size}_vp{vp_size}_ep{ep_size}_etp{etp_size}_{mbs}mbs_{gbs}gbs"  # pylint: disable=line-too-long
    exp_name = f"{task}_{model}_{model_size}_{args.compute_dtype}_{exp_config}"

    # configure executor
    if args.use_local_executor:
        assert (
            args.num_gpus == args.gpus_per_node
        ), "Local executor only supports running on a single node, make sure num_gpus is equal to gpus_per_node"

        executor = local_executor(
            args.gpu.lower(),
            args.num_gpus,
            custom_env_vars=kwargs.pop('custom_env_vars', {}),
            wandb_key=args.wandb_key,
            launcher="torchrun",
        )
    else:
        executor = slurm_executor(
            args.gpu.lower(),
            args.account,
            args.partition,
            args.log_dir,
            num_nodes,
            args.gpus_per_node,
            args.time_limit,
            args.container_image,
            custom_mounts=args.custom_mounts,
            custom_env_vars=kwargs.pop('custom_env_vars', {}),
            hf_token=args.hf_token,
            nemo_home=args.nemo_home,
            wandb_key=args.wandb_key,
            network='sharp' if args.use_sharp else None,
        )

    # configure plugins
    plugins = [build_perf_env_plugin(args, pp_size=pp_size, user_buffer_registration=use_user_buffer_registration)]
    if args.enable_nsys:
        plugins.append(
            NsysPlugin(
                start_step=kwargs.pop('nsys_start_step', 5),
                end_step=kwargs.pop('nsys_end_step', 6),
                ranks=kwargs.pop('nsys_ranks', [0]),
                gen_shape=kwargs.pop('nsys_gen_shape', False),
            )
        )
    if args.enable_memory_profile:
        assert args.memory_profile_out_path is not None
        plugins.append(MemoryProfilePlugin(dir=args.memory_profile_out_path))

    # run experiment
    if args.use_local_executor:
        if task != "pretrain" and (not kwargs.pop('finetuning_skip_import', True)):
            assert args.hf_token is not None, "HF token is required for importing checkpoint from HuggingFace"
            assert 'hf_model_url' in kwargs, "HF model URL is required for importing checkpoint from HuggingFace"
            run.run(*import_ckpt_experiment(executor, recipe.model, source=f"hf://{kwargs['hf_model_url']}"))
        run.run(recipe, executor=executor, plugins=plugins)
    else:
        with run.Experiment(exp_name) as exp:
            if task != "pretrain" and (not kwargs.pop('finetuning_skip_import', True)):
                assert args.hf_token is not None, "HF token is required for importing checkpoint from HuggingFace"
                assert 'hf_model_url' in kwargs, "HF model URL is required for importing checkpoint from HuggingFace"
                exp.add(*import_ckpt_experiment(executor, recipe.model, source=f"hf://{kwargs['hf_model_url']}"))

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

        # dump config diff from base recipe
        if args.dump_config_diff_from_base_recipe:
            output_dir = exp.jobs[-1].executor.job_dir
            # dump difference from base recipe
            base_recipe = kwargs.pop('base_recipe', None)
            if base_recipe is not None:
                file_name = f"diff_from_base_recipe_{args.compute_dtype}.diff"
                dump_config_diff_from_base_recipe(base_recipe, recipe, output_dir, file_name=file_name)
            # dump difference from default perf recipe
            default_perf_recipe = kwargs.pop('default_perf_recipe', None)
            if default_perf_recipe is not None:
                file_name = f"diff_from_default_perf_recipe_{args.compute_dtype}.diff"
                dump_config_diff_from_base_recipe(default_perf_recipe, recipe, output_dir, file_name=file_name)
