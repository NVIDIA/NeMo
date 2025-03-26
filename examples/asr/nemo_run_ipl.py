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

import datetime
import os
from pathlib import Path
from typing import Dict, List

import nemo_run as run
from omegaconf import OmegaConf, open_dict

from nemo.collections.common.parts import nemo_run_utils, ipl_utils
from examples.asr.nemo_run_helper import (
gather_mounts, 
check_root_path,
merge_configs, 
check_config_mount_paths,
update_exp_manager_runtime )
from nemo.core.config import hydra_runner
from nemo.utils import logging
NEMO_ROOT = Path(__file__).absolute().parents[2]


def get_pl_inference_command(inference_configs, shuffle=None):
    """
    Generate a command to run PL inference with multiple configuration files.
    Args:
        inference_configs (list): List of configuration file paths.

    Returns:
        str: Combined command string to execute PL inference.
    """
    # Base command template
    base_cmd = "python /nemo_run/code/examples/asr/transcribe_speech_parallel.py --config-path \"/results/configs\" --config-name {config_name}"
    if shuffle is not None:
        base_cmd += f" predict_ds.shuffle={shuffle}"

    # Generate the command list
    cmd_list = [base_cmd.format(config_name=os.path.basename(config)) for config in inference_configs]

    # Combine the commands with " && " separator
    return " && ".join(cmd_list)


def get_pseudo_labeling_command(
        merged_config: Dict, config_name: str, cluster_script_path: str, config_dir: str, ipl_training: Dict[str, any]) -> str:
    """
     Generate the pseudo-labeling command for the given configuration and training parameters.

    Args:
        merged_config (Dict): Merged configuration containing model and dataset settings.
        config_name (str): Name of the configuration file to be used.
        cluster_script_path (str): Path to the cluster execution script.
        config_dir (str): Directory containing the configuration files.
        ipl_training (Dict[str, any]): Dictionary containing:
            - first_run (bool): Whether this is the first run of pseudo-labeling.
            - num_gpus (int): Number of GPUs to use.
            - inference_config_paths (List[str]): List of inference configuration file paths.
            - manifests (List[str]): List of manifest file paths.
            - tarr_paths (List[str]): List of tarred audio file paths.
            - num_ipl_epochs (int): Number of epochs to train with pseudo-labels.
            - p_cache (float): What part of pseudo-labels to update.

    Returns:
        str: The constructed pseudo-labeling command.
    """
     
    prediction_directories_str = " ".join([os.path.dirname(path) for path in ipl_training['manifests']])
    inference_config_paths_str = " ".join(ipl_training['inference_config_paths'])

    updated_manifest_filepaths, updated_tarred_audio_filepaths = ipl_utils.update_training_sets(
        merged_config, ipl_training["manifests"], ipl_training.get("tarr_paths", None), ipl_training["prefix"]
    )

    exec_cmd = get_training_script_cmd(cluster_script_path, config_name)
    exec_cmd += " && sleep 10"
    if ipl_training.get("first_run", False):
        exec_cmd += f" && {get_pl_inference_command(ipl_training['inference_config_paths'], shuffle=False)}"
        exec_cmd += (
            f" && python /nemo_run/code/examples/asr/run_write_transcribed_files.py "
            f"--prediction_filepaths {prediction_directories_str} --full_pass --prefix {ipl_training['prefix']}"
        )
        if merged_config.model.train_ds.is_tarred:
            exec_cmd += " --is_tarred"
        exec_cmd += (
            f" && python /nemo_run/code/examples/asr/run_update_inf_config.py "
            f"--inference_configs {inference_config_paths_str} --p_cache {ipl_training['p_cache']} --num_gpus {ipl_training['num_gpus']}"
        )

    # If run has been interupted user has to change `num_ipl_epochs` in the config
    for _ in range(ipl_training["num_ipl_epochs"]):
        run_script = get_training_script_cmd(
            cluster_script_path, config_name, updated_manifest_filepaths, updated_tarred_audio_filepaths
        )
        exec_cmd += " && sleep 10"
        exec_cmd += f" && {run_script}"
        exec_cmd += f" && {get_pl_inference_command(ipl_training['inference_config_paths'],shuffle=True)}"
        exec_cmd += (
            f" && python /nemo_run/code/examples/asr/run_write_transcribed_files.py "
            f"--prediction_filepaths {prediction_directories_str} "
            f"--prefix {ipl_training['prefix']}"
        )
        if merged_config.model.train_ds.is_tarred:
            exec_cmd += " --is_tarred"

    return exec_cmd


def get_training_script_cmd(cluster_script_path, config_name, updated_manifest_filepaths=None, updated_tarred_filepaths=None):
    """
    Create the command to run the script on the cluster.

    Args:
        cluster_script_path (str): Path to the script to run on the cluster.
        config_name (str): Name of the config file to use for the script.
        updated_manifest_filepaths (str, optional): Path to the updated manifest file. Defaults to None.
        updated_tarred_filepaths (str, optional): Path to the updated tarred audio filepaths. Defaults to None.

    Returns:
        str: Command to run the script on the cluster.
    """

    # Prepare the base command for training
    cmd = (
        "find /results/ -name '*-unfinished' -type f -delete && "
        f"cd {os.path.dirname(cluster_script_path)} && "
        f"python -u -B {os.path.basename(cluster_script_path)} "
        f"--config-path \"/results/configs\" --config-name \"{config_name}\""
    )

    # Add additional parameters if provided
    if updated_manifest_filepaths:
        cmd += f" model.train_ds.manifest_filepath={updated_manifest_filepaths}"
    if updated_tarred_filepaths:
        cmd += f" model.train_ds.tarred_audio_filepaths={updated_tarred_filepaths}"

    return cmd

def get_export_variables_cmd(merged_cfg):
    wandb_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB") or os.environ.get("WANDB_KEY", "")
    if not wandb_key:
        logging.warning("WANDB key not found in environment variables. WANDB logging will not work.")

        # Check if WANDB logging is enabled in the exp_manager config
        if merged_cfg.get('exp_manager', {}).get('create_wandb_logger', False):
            raise ValueError(
                "WANDB key is required for logging but was not found in environment variables. "
                "Please set WANDB_API_KEY to enable WANDB logging."
            )

    cmd = (
        "nvidia-smi && "
        "export PYTHONPATH=/nemo_run/code && "
        f"export HF_TOKEN={os.getenv('HF_TOKEN', '')} && "
        f"export WANDB_API_KEY={wandb_key} && ")
    
    return cmd


@hydra_runner(config_path='conf', config_name='run_local')
def main(cluster_cfg):
    # Process the required arguments from the cluster config
    script_path = cluster_cfg.script
    script_config_path = cluster_cfg.script_config
    results_dir = cluster_cfg.results_dir

    script_path = Path(script_path).absolute()
    script_config_path = Path(script_config_path).absolute()

    # Gather all mounts from the cluster config; this includes any additional mounts provided by the user
    gather_mounts(cluster_cfg)

    # Add the results directory to the cluster config as a mount path
    nemo_run_utils.add_mount_path(results_dir, '/results', cluster_cfg)

    # Check if the script path is in the NeMo root directory
    cluster_script_path = check_root_path(script_path, NEMO_ROOT)

    # Create results and logdir
    log_dir = cluster_cfg.get('log_dir', os.path.join(results_dir, 'logs'))
    nemo_run_utils.create_remote_directory([results_dir, log_dir], cluster_cfg)

    # Load the script config and merge it with the cluster config
    script_config = OmegaConf.load(script_config_path)
    merged_config = merge_configs(script_config, cluster_cfg)

    # Update the exp_manager runtime with the max_runtime from the cluster config
    update_exp_manager_runtime(merged_config, cluster_cfg)

    # Perform all path checks in the merged config
    if "ipl_training" in merged_config.model:
        ipl_training = merged_config.model.ipl_training
        del merged_config.model.ipl_training
    else:
        raise KeyError("Parameters for `IPL` training are not provided.")
    
    check_config_mount_paths(merged_config, cluster_cfg)
    print()
    print(f"going to ipl part")

    
    print(f"ipl_training {ipl_training}")

    inference_config = ipl_training.inference_config
    inference_config_path = Path(inference_config).absolute()
    inference_config = OmegaConf.load(inference_config_path)

    # Resolve experiment name; if not provided in the script config file, check the cluster config
    exp_name = cluster_cfg.exp_name
    if exp_name is None:
        if 'exp_manager' in merged_config and 'name' in merged_config['exp_manager']:
            exp_name = merged_config['exp_manager']['name']
        else:
            raise ValueError(
                "Experiment name not provided in the run config file (`exp_name`)) or the cluster config (inside exp_manager.name)"
            )

    # Begin NeMo Run setup
    with run.Experiment(exp_name) as exp:
        # Create the config file name
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config_name = f"{exp_name}_{timestamp}_config.yaml"

        # Copy the merged config file to remote location's /results/configs directory
        config_dir = os.path.join(results_dir, 'configs')
        nemo_run_utils.create_remote_config(merged_config, config_name, config_dir, cluster_cfg)

        # Prepare arguments for the slurm job
        job_name = f"{exp_name}_job"

        # Get run parameters from the config
        num_runs = cluster_cfg.num_runs  # Number of dependent jobs for this script
        num_gpus = cluster_cfg.get('num_gpus', merged_config['trainer']['devices'])
        if isinstance(num_gpus, list):
            num_gpus = len(num_gpus)
        if num_gpus == -1:
            num_gpus = 1 if cluster_cfg['executor'] == 'local' else 8
            logging.warning(f"\n\nSetting num_gpus to {num_gpus} as it was set to -1\n\n")
        num_nodes = cluster_cfg.get('num_nodes', merged_config['trainer'].get('num_nodes', 1))


        checkpoint_dir = os.path.join(
            os.path.join(merged_config.exp_manager.exp_dir, merged_config.exp_manager.name), "checkpoints"
        )
        checkpoint_name = os.path.join(checkpoint_dir, merged_config.exp_manager.name + ".nemo")
        inference_config_paths, manifests, tarr_paths = nemo_run_utils.create_remote_inference_config(
            cluster_cfg, config_dir, inference_config, checkpoint_name
        )
        check_config_mount_paths(inference_config, cluster_cfg)
        # Add needed parameters for pseudo-labeling
        OmegaConf.set_struct(ipl_training, False)

        ipl_training['first_run'] = True
        ipl_training['num_gpus'] = num_nodes * num_gpus
        ipl_training['inference_config_paths'] = inference_config_paths
        ipl_training['manifests'] = manifests
        ipl_training['tarr_paths'] = tarr_paths

        cmd = get_pseudo_labeling_command(
            merged_config, config_name, cluster_script_path, config_dir, ipl_training
        )

        # Cast the cluster config to a dictionary for compatibility with NeMo Run
        cluster_cfg = OmegaConf.to_object(cluster_cfg)

        logging.info(f"Scheduling {num_runs} runs of the script {script_path}...")
        task = None
        for run_id in range(num_runs):
            # Add the task to the experiment
            if run_id == 0:
                task = None
            else:
                if ipl_training:
                    ipl_training['first_run'] = False
                    cmd = get_pseudo_labeling_command(
                        merged_config, config_name, cluster_script_path, config_dir, ipl_training
                    )
                task = [task]

            task = nemo_run_utils.add_task(
                exp,
                cmd=cmd,
                task_name=job_name,
                cluster_config=cluster_cfg,
                container=cluster_cfg['containers']['asr'],
                num_tasks=cluster_cfg.get('num_tasks', cluster_cfg.get('num_tasks_per_node', 1)),
                num_gpus=num_gpus,
                num_nodes=num_nodes,
                log_dir=nemo_run_utils.get_mounted_filepath(cluster_cfg, log_dir),
                partition=cluster_cfg.get('partition', None),
                task_dependencies=task,
            )

        # Run the experiment on the cluster with all the tasks
        nemo_run_utils.run_exp(exp, cluster_cfg)


if __name__ == '__main__':
    main()
