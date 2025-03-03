# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.common.parts import run_ipl_utils, run_utils
from nemo.core.config import hydra_runner
from nemo.utils import logging

NEMO_ROOT = Path(__file__).absolute().parents[2]


def gather_mounts(cluster_cfg):
    """
    Gather all mounts from the cluster config including ones which are disjoint from the cluster_cfg.mounts list.
    It is used because Hydra does not support the ability to append to a list in the config file natively.

    Users can provide additional mounts from the command line using the following syntax:
    ++mount_<anything>='/src:/dest'

    Args:
        cluster_cfg: Cluster config dictionary
    """
    # Gather all mounts from the cluster config including ones which are disjoint from the cluster_cfg.mounts list.
    mounts = cluster_cfg.get('mounts', [])

    # Resolve any mounts in th cluster config that need user expansion
    mounts = [os.path.expanduser(m) for m in mounts]

    keys = list(cluster_cfg.keys())
    # Check for any additional mounts in the cluster config
    with open_dict(cluster_cfg):
        for k in keys:
            if k.startswith("mount_"):  # Additional mount found
                logging.info(f"Found additional mount flag in the cluster config `{k}`. Adding it to the mounts list.")
                mounts.append(cluster_cfg[k])
                del cluster_cfg[k]  # Remove the key from the cluster config

        cluster_cfg['mounts'] = mounts
        logging.info(f"Final Mounts: {mounts}")


def check_root_path(path, nemo_root):
    """
    Check if a path is in the NeMo root directory and convert it to a path that is relative to the NeMo root directory.
    This is used to ensure that any path that is provided to this script will be in the NeMo root directory when
    mounted in the container.

    Args:
        path: Path to check
        nemo_root: NeMo root directory

    Returns:
        str: Path relative to the NeMo root directory
    """
    path = str(path)
    nemo_root = str(nemo_root)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist.")

    if not path.startswith(nemo_root):
        raise ValueError(f"Path {path} is not in the NeMo root directory.")

    new_path = path.replace(nemo_root, '/nemo_run/code/')
    return new_path


def merge_configs(script_config, cluster_cfg):
    """
    Merge the script config and the cluster config and resolve the final values.
    The script config will take precedence over the cluster config.

    **Note**: The script config will NOT be resolved - it will maintain its hydra placeholders for resolution at
    runtime on the cluster.

    Args:
        script_config: Script config dictionary that represents the Model training/inference config
        cluster_cfg: Cluster config dictionary that represents the cluster configuration

    Returns:
        dict: Merged config dictionary
    """
    original_script_keys = set(script_config.keys())

    # Copy the cluster config and resolve it to get the final values before merging
    run_copy = OmegaConf.masked_copy(cluster_cfg, keys=list(cluster_cfg.keys()))
    OmegaConf.resolve(run_copy)
    result = OmegaConf.merge(script_config, run_copy)

    # Delete cluster config keys from the merged config
    with open_dict(result):
        for k in cluster_cfg.keys():
            if k in result and k not in original_script_keys:
                del result[k]

    # Check for any ??? missing values in result recursively and raise an error if found
    def check_missing_values(cfg):
        if hasattr(cfg, 'items'):
            for k, v in cfg.items():
                if hasattr(v, 'items'):
                    check_missing_values(v)
                elif v == '???':
                    raise ValueError(f"Missing value for key {k} in the config file")

    check_missing_values(result)

    # Do name check as a special case
    if 'name' in result and result['name'] == '':
        raise ValueError(
            f"Missing value for key 'name' in the merged config file (value={result['name']}).\n"
            f"Check if your ++ override is using single quote (') instead of double quote (\") for resolution.\n"
            "Example: ++name='${exp_name}'"
        )

    return result


def check_config_mount_paths(script_config, cluster_config):
    """
    Check if all path-like strings in the script config are mounted paths in the cluster config.
    If a path-like string is not a mounted path, raise an error.

    Args:
        script_config: Script config dictionary that represents the Model training/inference config
        cluster_config: Cluster config dictionary that represents the cluster configuration
    """
    # recursively walk all values of the script_config, checking if its a path-like string and if so, check if the path is a mounted path
    # if it is not, raise an error

    if 'AIS_ENDPOINT' in os.environ:
        ais_endpoint = os.environ['AIS_ENDPOINT']
    else:
        ais_endpoint = None

    # Check if ais paths should be checked at all
    # This can be disabled using `++check_ais_paths=False` passed when calling the script
    check_ais_paths = cluster_config.get('check_ais_paths', True)

    def filepath_check(v, cluster_cfg):
        if v.startswith(os.path.sep):  # check for absolute paths only
            logging.info(f"Checking if {v} is a mounted path")
            # Check if the path begins with mount path
            run_utils.check_if_mounted(cluster_cfg, v)

            # Check the file exists in the cluster at the unmounted path
            unmounted_path = run_utils.get_unmounted_filepath(cluster_cfg, v)
            run_utils.check_remote_mount_directories(unmounted_path, cluster_cfg)

        elif (
            check_ais_paths and "ais://" in v and ais_endpoint is not None
        ):  # if the value is a string, check if its an ais path
            # Try to import ais module
            try:
                from aistore.sdk import Client

                # Do actual data check for this ais path
                ais_client = Client(ais_endpoint)
                ais_client.fetch_object_by_url(v).head()

            except ImportError:
                logging.warning("\nais module is not installed. Please install it to use ais paths.\n")

    def check_mounted_path(cfg, cluster_cfg):
        if hasattr(cfg, 'items'):  # if the object is a dictionary
            for k, v in cfg.items():
                if hasattr(v, 'items'):  # if the value is a dictionary, recurse
                    check_mounted_path(v, cluster_cfg)

                elif isinstance(v, list):  # if the value is a list, check if its items are an absolute path
                    for item in v:
                        if isinstance(item, str):
                            filepath_check(item, cluster_cfg)

                elif isinstance(v, str):  # if the value is a string, check if its an absolute a path
                    filepath_check(v, cluster_cfg)

    check_mounted_path(script_config, cluster_config)


def update_exp_manager_runtime(script_config, cluster_cfg):
    """
    Update the max_time_per_run in the exp_manager config in the script config with the max_runtime from the cluster config.

    Args:
        script_config: Script config dictionary that represents the Model training/inference config
        cluster_cfg: Cluster config dictionary that represents the cluster configuration
    """
    if 'max_runtime' in cluster_cfg:
        with open_dict(script_config):
            if 'exp_manager' not in script_config:
                raise ValueError("exp_manager config not found in the script config file")

            script_config['exp_manager']['max_time_per_run'] = cluster_cfg['max_runtime']
            logging.info(f"Setting exp_manager.max_time_per_run to {cluster_cfg['max_runtime']}")


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

    updated_manifest_filepaths, updated_tarred_audio_filepaths = run_ipl_utils.update_training_sets(
        merged_config, ipl_training["manifests"], ipl_training.get("tarr_paths", None), ipl_training["prefix"]
    )

    exec_cmd = get_execution_script(cluster_script_path, config_name, config_dir)
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

    for _ in range(ipl_training["num_ipl_epochs"]):
        run_script = get_execution_script(
            cluster_script_path, config_name, config_dir, updated_manifest_filepaths, updated_tarred_audio_filepaths
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
    #exec_cmd = "python /nemo_run/code/examples/asr/run_update_inf_config.py "
    return exec_cmd


def get_execution_script(
    cluster_script_path, config_name, merged_cfg, updated_manifest_filepaths=None, updated_tarred_filepaths=None
):
    """
    Create the command to run the script on the cluster.

    Args:
        cluster_script_path (str): Path to the script to run on the cluster.
        config_name (str): Name of the config file to use for the script.
        merged_cfg (dict): Merged config dictionary representing the model training/inference configuration.
        updated_manifest_filepaths (str, optional): Path to the updated manifest file. Defaults to None.
        updated_tarred_filepaths (str, optional): Path to the updated tarred audio filepaths. Defaults to None.

    Returns:
        str: Command to run the script on the cluster.
    """
    # Get the WANDB API key from the environment variables
    wandb_key = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB") or os.environ.get("WANDB_KEY", "")
    if not wandb_key:
        logging.warning("WANDB key not found in environment variables. WANDB logging will not work.")

        # Check if WANDB logging is enabled in the exp_manager config
        if merged_cfg.get('exp_manager', {}).get('create_wandb_logger', False):
            raise ValueError(
                "WANDB key is required for logging but was not found in environment variables. "
                "Please set WANDB_API_KEY to enable WANDB logging."
            )

    # Prepare the base command
    cmd = (
        "nvidia-smi && "
        "export PYTHONPATH=/nemo_run/code && "
        f"export HF_TOKEN={os.getenv('HF_TOKEN', '')} && "
        f"export WANDB_API_KEY={wandb_key} && "
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

    # Add fallback directory listing
    if not updated_manifest_filepaths and not updated_tarred_filepaths:
        cmd += " && cd /results && ls -l"

    return cmd


@hydra_runner(config_path='conf', config_name='run_local')
def main(cluster_cfg):
    # Process the required arguments from the cluster config
    script_path = cluster_cfg.script
    script_config_path = cluster_cfg.script_config
    results_dir = cluster_cfg.results_dir

    script_path = Path(script_path).absolute()
    script_config_path = Path(script_config_path).absolute()

    ipl_training = cluster_cfg.get("ipl_training", None)

    if ipl_training:
        inference_config = cluster_cfg.ipl_training.inference_config
        inference_config_path = Path(inference_config).absolute()
        inference_config = OmegaConf.load(inference_config_path)

    # Gather all mounts from the cluster config; this includes any additional mounts provided by the user
    gather_mounts(cluster_cfg)

    # Add the results directory to the cluster config as a mount path
    run_utils.add_mount_path(results_dir, '/results', cluster_cfg)

    # Check if the script path is in the NeMo root directory
    cluster_script_path = check_root_path(script_path, NEMO_ROOT)

    # Create results and logdir
    log_dir = cluster_cfg.get('log_dir', os.path.join(results_dir, 'logs'))
    run_utils.create_remote_directory([results_dir, log_dir], cluster_cfg)

    # Load the script config and merge it with the cluster config
    script_config = OmegaConf.load(script_config_path)
    merged_config = merge_configs(script_config, cluster_cfg)

    # Update the exp_manager runtime with the max_runtime from the cluster config
    update_exp_manager_runtime(merged_config, cluster_cfg)

    # Perform all path checks in the merged config
    check_config_mount_paths(merged_config, cluster_cfg)

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

        # Get the execution script
        cmd = get_execution_script(cluster_script_path, config_name, merged_config, cluster_cfg)

        # Copy the merged config file to remote location's /results/configs directory
        config_dir = os.path.join(results_dir, 'configs')
        run_utils.create_remote_config(merged_config, config_name, config_dir, cluster_cfg)

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

        if not ipl_training:
            cmd = get_execution_script(cluster_script_path, config_name, merged_config, cluster_cfg)
        else:
            checkpoint_dir = os.path.join(
                os.path.join(merged_config.exp_manager.exp_dir, merged_config.exp_manager.name), "checkpoints"
            )
            checkpoint_name = os.path.join(checkpoint_dir, merged_config.exp_manager.name + ".nemo")
            inference_config_paths, manifests, tarr_paths = run_utils.create_remote_inference_config(
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

            task = run_utils.add_task(
                exp,
                cmd=cmd,
                task_name=job_name,
                cluster_config=cluster_cfg,
                container=cluster_cfg['containers']['asr'],
                num_tasks=cluster_cfg.get('num_tasks', cluster_cfg.get('num_tasks_per_node', 1)),
                num_gpus=num_gpus,
                num_nodes=num_nodes,
                log_dir=run_utils.get_mounted_filepath(cluster_cfg, log_dir),
                partition=cluster_cfg.get('partition', None),
                task_dependencies=task,
            )

        # Run the experiment on the cluster with all the tasks
        run_utils.run_exp(exp, cluster_cfg)


if __name__ == '__main__':
    main()
