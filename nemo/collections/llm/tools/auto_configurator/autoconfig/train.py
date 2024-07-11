# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""Module to launch training jobs using nemo_framework_launcher."""

import os
import subprocess

from autoconfig import utils
from omegaconf import OmegaConf


def run_training(
    file_name: str, model_name: str, results_dir: str, cfg: OmegaConf
) -> int:
    """
    Launch a training job for the given model name and config file, using nemo_framework_launcher.
    :param str file_name: name of the file configuration to be selected for training with nemo_framework_launcher.
    :param str model_name: model type to be run, usually gpt3, t5 or mt5.
    :param str results_dir: path to the directory where the results will be stored.
    :param OmegaConf cfg: OmegaConf object with full configuration for the HP tool.
    :return: SLURM job_id of the training job that was launched.
    :rtype: str
    """
    # Copy cluster config to nemo_framework_launcher.
    launcher_scripts_path = cfg.get("launcher_scripts_path")
    cluster_cfg = cfg.get("cluster")
    dst = os.path.join(launcher_scripts_path, "conf/cluster/bcm.yaml")
    copy_config_to_file(cluster_cfg, dst)
    print(f"Copied cluster config to {dst}")

    # Generate string of hydra overrides for nemo_framework_launcher.
    overrides_str = generate_overrides_str(file_name, model_name, results_dir, cfg)

    nemo_megatron_ci = (
        f"NEMO_LAUNCHER_CI=1" if bool(os.getenv("NEMO_LAUNCHER_CI")) else ""
    )
    main_path = os.path.join(launcher_scripts_path, "main.py")
    cmd = f"HYDRA_FULL_ERROR=1 {nemo_megatron_ci} python3 {main_path} {overrides_str} "

    # Launch job with command cmd.
    try:
        job_output = subprocess.check_output([cmd], shell=True).decode("utf-8")
        job_id = job_output.split(" ")[-1]
    except Exception as err:
        job_id = None
        print(err)
    print(f"Submitted Training script with job id: {job_id}")
    return job_id


def copy_config_to_file(cfg: OmegaConf, dst: str) -> None:
    """
    Copies OmegaConf configuration to a dst file.
    :param OmegaConf cfg: OmegaConfg object with the config to be stored in a file.
    :param str dst: destination path to where the config will be stored. Must be a yaml file.
    :return: None
    """
    with open(dst, "w") as f:
        OmegaConf.save(config=cfg, f=f)


def convert_to_absolute_path(path: str) -> str:
    """
    Removes the /../ part from relative paths to convert them to absolute paths.
    :param str path: the path that will be converted.
    :return: the converted path with no /../ elements in it.
    :rtype: str
    """
    path_split = path.split("/")
    result = []
    for index, elem in enumerate(path_split):
        if elem == "..":
            result.pop(-1)
        else:
            result.append(elem)
    return "/".join(result)


def generate_overrides_str(
    file_name: str, model_name: str, results_dir: str, cfg: OmegaConf
) -> str:
    """
    Generates string with hydra-like parameter overrides for nemo_framework_launcher.
    :param str file_name: name of the file configuration to be selected for training with nemo_framework_launcher.
    :param str model_name: model type to be run, usually gpt3, t5 or mt5.
    :param str results_dir: path to the directory where the results will be stored.
    :param OmegaConf cfg: OmegaConf object with full configuration for the HP tool.
    :return: string containing all the hydra-like overrides required for the training job.
    :rtype: str
    """
    file_name = file_name.replace(".yaml", "")
    training_model = f"{model_name}/{file_name}"
    cluster_type = cfg.get("cluster_type")
    container = cfg.get("training_container")
    auto_configurator_path = cfg.get("auto_configurator_path")
    auto_configurator_path = convert_to_absolute_path(auto_configurator_path)
    launcher_scripts_path = cfg.get("launcher_scripts_path")
    launcher_scripts_path = convert_to_absolute_path(launcher_scripts_path)
    data_dir = cfg.get("data_dir")
    container_mounts = cfg.get("container_mounts", "null")
    api_key_file = cfg.get("wandb").get("api_key_file")
    if api_key_file is None:
        api_key_file = "null"

    # Process container-mounts.
    mounts_str = (
        f"{auto_configurator_path}:{auto_configurator_path},{results_dir}:{results_dir}"
    )
    mounts_str += utils.add_container_mounts(container_mounts)

    overrides_str = (
        f"training={training_model} "
        f"stages=[training] "
        f"cluster_type={cluster_type} "
        f"base_results_dir={results_dir} "
        f"\"container='{container}'\" "
        f"launcher_scripts_path={launcher_scripts_path} "
        f"data_dir={data_dir} "
        f"training.exp_manager.create_checkpoint_callback=False "
        f"container_mounts=\[{mounts_str}\] "
        f"wandb_api_key_file={api_key_file} "
    )
    return overrides_str
