import sys
import os
import copy
import subprocess

import hydra
import omegaconf
import functools
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Iterable

from bignlp.core.launchers import AutoLauncher
from bignlp.core.stages import BigNLPStage
from bignlp.core.stages import clean_command_groups, create_args_list
from bignlp.utils.job_utils import JobPaths


class Export(BigNLPStage):

    def setup_stage_vars(self, cfg):
        self.stage_name = "export"
        self.stage_cfg = cfg.get("export")
        self.accuracy_cfg = self.stage_cfg.get("accuracy")

    def _make_checkpoint_search_command(self, **kwargs):
        checkpoint_override = [f"{k}={v}" for k, v in kwargs.items()]
        return (
            f"python3 {self._bignlp_path / 'bignlp/collections/checkpoint_search.py'} "
            f"{' '.join(checkpoint_override)}"
        )

    def make_stage_command_groups(
            self, stage_cfg_path, sub_stage=None,
    ) -> List[List[str]]:
        command_groups = [[]]

        command_groups[0] += self._make_sub_stage_command(sub_stage)
        command_groups = clean_command_groups(command_groups)
        return command_groups
    
    def _make_sub_stage_command(self, sub_stage):
        choice_model_type = self.cfg.get(f"{self.stage_name}_config")
        cmds_fn = {
            "convert": {
                "gpt3": self._get_gpt_conversion_cmds,
                "t5": self._get_t5_conversion_cmds,
                "mt5": self._get_t5_conversion_cmds,
            },
            "accuracy": {
                "gpt3": self._get_gpt_accuracy_cmds,
                "t5": self._get_t5_accuracy_cmds,
                "mt5": self._get_mt5_accuracy_cmds,
            },
        }[sub_stage][choice_model_type]
        return cmds_fn(self.cfg)

    def _make_sub_stages(self):
        sub_stages = ["convert"]
        if self.accuracy_cfg.get("enabled", False):
            sub_stages.append("accuracy")
        return sub_stages
    
    def setup_folder_and_data(self) -> None:
        """Setup required folders and dataset"""
        job_path = self.get_job_path()
        job_path.folder.mkdir(parents=True, exist_ok=True)
        results_folder = job_path.results_folder
        results_folder.mkdir(parents=True, exist_ok=True)

    def run(self) -> str:
        # Setup folders and datasets
        self.setup_folder_and_data()

        sub_stages = self._make_sub_stages()
        job_id = ""
        for sub_stage in sub_stages:
            # Save stage hydra config
            job_path = self.get_job_path(sub_stage)
            job_path.folder.mkdir(parents=True, exist_ok=True)

            stage_cfg_path = BigNLPStage.save_stage_hydra_config(
                self.stage_cfg, job_path
            )
            if job_id:
                dependency = f"aftercorr:{job_id}"
                self.stage_cfg["run"]["dependency"] = dependency

            # Make cluster parameters
            cluster_parameters = self._make_cluster_parameters(self.cluster, sub_stage)

            # Make command groups
            command_groups = self.make_stage_command_groups(stage_cfg_path, sub_stage)
            # Create launcher
            launcher = AutoLauncher(
                folder=job_path.folder,
                cluster=self.cluster,
                **cluster_parameters,
            )
            job_id = launcher.launch(command_groups=command_groups)

        return job_id
    
    def _make_cluster_parameters(
            self, cluster: str, sub_stage=None,
    ) -> Dict:
        cfg = self.cfg
        stage_cfg = self.stage_cfg

        run_cfg = stage_cfg.get("run")
        job_name = run_cfg.get("name")
        time_limit = run_cfg.get("time_limit")
        dependency = run_cfg.get("dependency")

        env_vars = self.get_env_vars()
        env_vars["PYTHONPATH"] = f"{self._bignlp_path}:${{PYTHONPATH}}" # Required by pile download
        env_vars["NGC_ARRAY_TYPE"] = "MPIJob" # Required by BCP
        setup = [
            f"export {k}={v}" for k, v in env_vars.items()
        ]

        cluster_parameters = {}
        shared_parameters = {
            "job_name": job_name,
            "time": time_limit,
            "setup": setup,
        }
        private_parameters = self._make_private_cluster_parameters(
            cluster, sub_stage,
        )
        if cluster == "bcm":
            cluster_cfg = cfg.get("cluster")
            slurm_cfg = {**copy.deepcopy(cluster_cfg)}
            job_name_prefix = slurm_cfg.pop("job_name_prefix")
            cluster_parameters = {
                **slurm_cfg,
                "dependency": dependency,
            }
            cluster_parameters.update({
                **shared_parameters,
                **private_parameters,
            })
            cluster_parameters["job_name"] = job_name_prefix + cluster_parameters["job_name"]
        elif cluster == "bcp":
            cluster_parameters.update({
                **shared_parameters,
                **private_parameters,
            })
        elif cluster == "interactive":
            cluster_parameters.update(shared_parameters)
        return cluster_parameters

    def _make_private_cluster_parameters(self, cluster, sub_stage):
        cfg = self.cfg
        stage_cfg = self.stage_cfg
        run_cfg = stage_cfg.get("run")
        accuracy_cfg = stage_cfg.get("accuracy")
        ft_model_cfg = stage_cfg.get("model")
        triton_cfg = stage_cfg.get("triton_deployment")

        container_image = cfg.get("container")
        container_mounts = self._make_container_mounts_string()

        num_tasks = ft_model_cfg.tensor_model_parallel_size * triton_cfg.pipeline_model_parallel_size
        nodes = 1 if sub_stage == "export" else int(math.ceil(num_tasks / accuracy_cfg.ntasks_per_node))
        ntasks_per_node = 1 if sub_stage == "export" else accuracy_cfg.ntasks_per_node

        if cluster == "bcm":
            return {
                "nodes": nodes,
                "ntasks_per_node": ntasks_per_node,
                "container_image": container_image,
                "container_mounts": container_mounts,
            }
        if cluster == "bcp":
            return {
                "nodes": nodes,
                "ntasks_per_node": ntasks_per_node,
                "bcp_launcher": "'mpirun --allow-run-as-root'",
            }
        return {}

    def _get_gpt_conversion_cmds(self, cfg):
        run_cfg = cfg.export.run
        ft_model_cfg = cfg.export.model
        triton_cfg = cfg.export.triton_deployment
        tokenizer_cfg = cfg.training.model.tokenizer

        checkpoint_path = ft_model_cfg.checkpoint_path
        triton_model_dir = triton_cfg.triton_model_dir

        bignlp_scripts_path = pathlib.Path(cfg.bignlp_path)
        converter_path = FT_PATH / "examples/pytorch/gpt/utils/nemo_ckpt_convert.py"
        prepare_model_config_script_path = bignlp_scripts_path / "bignlp/export_scripts/prepare_triton_model_config.py"
        template_path = FT_BACKEND_PATH / "all_models/gpt/fastertransformer/config.pbtxt"

        triton_model_version_dir = f"{triton_model_dir}/1"

        convert_cmd = (
            f"python -u {converter_path} \\\n"
            f" --in-file {checkpoint_path} \\\n"
            f" --saved-dir {triton_model_version_dir} \\\n"
            f" --infer-gpu-num {ft_model_cfg.tensor_model_parallel_size} \\\n"
            f" --weight-data-type {ft_model_cfg.weight_data_type} \\\n"
            f" --vocab-path {tokenizer_cfg.vocab_file} \\\n"
            f" --merges-path {tokenizer_cfg.merge_file} \\\n"
            f" --processes {ft_model_cfg.processes} \\\n"
            f" --load-checkpoints-to-cpu {int(ft_model_cfg.load_checkpoints_to_cpu)}"
        )
        triton_prepare_model_config_cmd = (
            f"python -u {prepare_model_config_script_path} \\\n"
            f" --model-train-name {run_cfg.model_train_name} \\\n"
            f" --template-path {template_path} \\\n"
            f" --ft-checkpoint {triton_model_version_dir}/{ft_model_cfg.tensor_model_parallel_size}-gpu \\\n"
            f" --config-path {triton_model_dir}/config.pbtxt \\\n"
            f" --max-batch-size {triton_cfg.max_batch_size} \\\n"
            f" --pipeline-model-parallel-size {triton_cfg.pipeline_model_parallel_size} \\\n"
            f" --tensor-model-parallel-size {ft_model_cfg.tensor_model_parallel_size} \\\n"
            f" --data-type {triton_cfg.data_type}"
        )
        if triton_cfg.int8_mode:
            triton_prepare_model_config_cmd += " \\\n --int8-mode"
        if triton_cfg.enable_custom_all_reduce:
            triton_prepare_model_config_cmd += " \\\n --enable-custom-all-reduce"
        return [
            (
                f"export PYTHONPATH={FT_PATH}:${{PYTHONPATH}} && \\\n"
                + f"rm -rf {triton_model_dir} && \\\n"  # to not mix old and newly generated FT checkpoint files
                + f"{convert_cmd} && \\\n"
                + triton_prepare_model_config_cmd
            )
        ]

    def _get_t5_conversion_cmds(self, cfg):
        run_cfg = cfg.export.run
        ft_model_cfg = cfg.export.model
        triton_cfg = cfg.export.triton_deployment

        checkpoint_path = ft_model_cfg.checkpoint_path
        triton_model_dir = triton_cfg.triton_model_dir

        bignlp_scripts_path = pathlib.Path(cfg.bignlp_path)
        converter_path = FT_PATH / "examples/pytorch/t5/utils/nemo_t5_ckpt_convert.py"
        prepare_model_config_script_path = bignlp_scripts_path / "bignlp/export_scripts/prepare_triton_model_config.py"
        template_path = FT_BACKEND_PATH / "all_models/t5/fastertransformer/config.pbtxt"

        triton_model_version_dir = f"{triton_model_dir}/1"

        convert_cmd = (
            f"python -u {converter_path} \\\n"
            f" --in-file {checkpoint_path} \\\n"
            f" --saved-dir {triton_model_version_dir} \\\n"
            f" --model-name {run_cfg.model_train_name} \\\n"
            f" --infer-gpu-num {ft_model_cfg.tensor_model_parallel_size} \\\n"
            f" --weight-data-type {ft_model_cfg.weight_data_type} \\\n"
            f" --processes {ft_model_cfg.processes}"
        )
        triton_prepare_model_config_cmd = (
            f"python -u {prepare_model_config_script_path} \\\n"
            f" --model-train-name {run_cfg.model_train_name} \\\n"
            f" --template-path {template_path} \\\n"
            f" --ft-checkpoint {triton_model_version_dir}/{ft_model_cfg.tensor_model_parallel_size}-gpu \\\n"
            f" --config-path {triton_model_dir}/config.pbtxt \\\n"
            f" --max-batch-size {triton_cfg.max_batch_size} \\\n"
            f" --pipeline-model-parallel-size {triton_cfg.pipeline_model_parallel_size} \\\n"
            f" --tensor-model-parallel-size {ft_model_cfg.tensor_model_parallel_size} \\\n"
            f" --data-type {triton_cfg.data_type}"
        )
        if triton_cfg.int8_mode:
            triton_prepare_model_config_cmd += " \\\n --int8-mode"
        if triton_cfg.enable_custom_all_reduce:
            triton_prepare_model_config_cmd += " \\\n --enable-custom-all-reduce"
        return [
            (
                f"export PYTHONPATH={FT_PATH}:${{PYTHONPATH}} && \\\n"
                + f"rm -rf {triton_model_dir} && \\\n"  # to not mix old and newly generated FT checkpoint files
                + f"{convert_cmd} && \\\n"
                + triton_prepare_model_config_cmd
            )
        ]

    def _get_gpt_accuracy_cmds(self, cfg):
        run_cfg = cfg.export.run
        ft_model_cfg = cfg.export.model
        triton_cfg = cfg.export.triton_deployment
        accuracy_cfg = cfg.export.accuracy

        checkpoint_path = f"{triton_cfg.triton_model_dir}/1/{ft_model_cfg.tensor_model_parallel_size}-gpu"

        lambada_script_path = FT_PATH / "examples/pytorch/gpt/lambada_task_example.py"
        update_config_script_path = FT_PATH / "examples/pytorch/gpt/utils/update_gpt_config.py"
        lib_path = FT_PATH_WITH_BUILD / "build/lib/libth_parallel_gpt.so"

        lambada_path = accuracy_cfg.test_data
        create_config_ini_cmd = (
            f"mkdir -p $(dirname {accuracy_cfg.runtime_config_ini_path}) && \\\n"
            f"cp {checkpoint_path}/config.ini {accuracy_cfg.runtime_config_ini_path} && \\\n"
            f"python -u {update_config_script_path} \\\n"
            f" --model-dir {checkpoint_path} \\\n"
            f" --config-ini-path {accuracy_cfg.runtime_config_ini_path} \\\n"
            f" --pipeline-para-size {triton_cfg.pipeline_model_parallel_size} \\\n"
            f" --tensor-para-size {ft_model_cfg.tensor_model_parallel_size} \\\n"
            f" --max-seq-len {accuracy_cfg.runtime.max_seq_len} \\\n"
            f" --beam-width {accuracy_cfg.runtime.beam_width} \\\n"
            f" --sampling-top-k {accuracy_cfg.runtime.sampling_top_k} \\\n"
            f" --sampling-top-p {accuracy_cfg.runtime.sampling_top_p} \\\n"
            f" --data-type {triton_cfg.data_type} && \\\n"
            f"mkdir -p $(dirname {lambada_path}) && \\\n"
            f"wget https://github.com/cybertronai/bflm/raw/master/lambada_test.jsonl -O {lambada_path}"
        )

        lambada_cmd = (
            f"python -u {lambada_script_path} \\\n"
            f" --checkpoint-path {checkpoint_path} \\\n"
            f" --lib-path {lib_path} \\\n"
            f" --config-ini-path {accuracy_cfg.runtime_config_ini_path} \\\n"
            f" --lambada-path {lambada_path} \\\n"
            f" --output-path {accuracy_cfg.output_path} \\\n"
            f" --batch-size {accuracy_cfg.batch_size}"
        )

        # separate into 2 tasks to not start lambada cmd before configurations and data files are prepared
        # LOCAL_RANK is set with an enroot hook for Pytorch containers
        # SLURM_LOCALID is set by Slurm
        # OMPI_COMM_WORLD_LOCAL_RANK is set by mpirun
        return [
            (
                f"export PYTHONPATH={FT_PATH}:${{PYTHONPATH}} && \\\n"
                'export MY_LOCAL_RANK="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}" && \\\n'
                'if [ ${MY_LOCAL_RANK} = "0" ]; then ' + create_config_ini_cmd + "; fi"
            ),
            f"export PYTHONPATH={FT_PATH}:${{PYTHONPATH}} && \\\n" + lambada_cmd,
        ]

    def _get_t5_accuracy_cmds(self, cfg):
        run_cfg = cfg.export.run
        ft_model_cfg = cfg.export.model
        triton_cfg = cfg.export.triton_deployment
        accuracy_cfg = cfg.export.accuracy

        checkpoint_path = f"{triton_cfg.triton_model_dir}/1/{ft_model_cfg.tensor_model_parallel_size}-gpu"

        mnli_script_path = FT_PATH / "examples/pytorch/t5/mnli_task_example.py"
        lib_path = FT_PATH_WITH_BUILD / "build/lib/libth_t5.so"
        mnli_path = accuracy_cfg.test_data

        mnli_cmd = (
            f"python -u {mnli_script_path} \\\n"
            f" --ckpt_path {checkpoint_path} \\\n"
            f" --lib_path {lib_path} \\\n"
            f" --data_path {mnli_path} \\\n"
            f" --output_path {accuracy_cfg.output_path} \\\n"
            f" --batch_size {accuracy_cfg.batch_size} \\\n"
            f" --max_output_len {accuracy_cfg.max_output_len} \\\n"
            f" --beam_width {accuracy_cfg.runtime.beam_width} \\\n"
            f" --sampling_topk {accuracy_cfg.runtime.sampling_top_k} \\\n"
            f" --sampling_topp {accuracy_cfg.runtime.sampling_top_p} \\\n"
            f" --data_type {triton_cfg.data_type} \\\n"
            f" --tensor_para_size {ft_model_cfg.tensor_model_parallel_size} \\\n"
            f" --pipeline_para_size {triton_cfg.pipeline_model_parallel_size}"
        )

        return [
            f"export PYTHONPATH={FT_PATH}:${{PYTHONPATH}} && \\\n" + mnli_cmd,
        ]

    def _get_mt5_accuracy_cmds(self, cfg):
        run_cfg = cfg.export.run
        ft_model_cfg = cfg.export.model
        triton_cfg = cfg.export.triton_deployment
        accuracy_cfg = cfg.export.accuracy

        checkpoint_path = f"{triton_cfg.triton_model_dir}/1/{ft_model_cfg.tensor_model_parallel_size}-gpu"

        xnli_script_path = FT_PATH / "examples/pytorch/t5/xnli_task_example.py"
        lib_path = FT_PATH_WITH_BUILD / "build/lib/libth_t5.so"
        xnli_path = accuracy_cfg.test_data

        xnli_cmd = (
            f"python -u {xnli_script_path} \\\n"
            f" --ckpt_path {checkpoint_path} \\\n"
            f" --lib_path {lib_path} \\\n"
            f" --data_path {xnli_path} \\\n"
            f" --output_path {accuracy_cfg.output_path} \\\n"
            f" --batch_size {accuracy_cfg.batch_size} \\\n"
            f" --max_output_len {accuracy_cfg.max_output_len} \\\n"
            f" --beam_width {accuracy_cfg.runtime.beam_width} \\\n"
            f" --sampling_topk {accuracy_cfg.runtime.sampling_top_k} \\\n"
            f" --sampling_topp {accuracy_cfg.runtime.sampling_top_p} \\\n"
            f" --data_type {triton_cfg.data_type} \\\n"
            f" --tensor_para_size {ft_model_cfg.tensor_model_parallel_size} \\\n"
            f" --pipeline_para_size {triton_cfg.pipeline_model_parallel_size}"
        )

        return [
            f"export PYTHONPATH={FT_PATH}:${{PYTHONPATH}} && \\\n" + xnli_cmd,
        ]
