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

import os
import datetime
from pathlib import Path

import nemo_run as run
from omegaconf import OmegaConf, open_dict

from nemo.collections.common.parts import run_utils
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
        raise ValueError(f"Missing value for key 'name' in the merged config file (value={result['name']}).\n"
                         f"Check if your ++ override is using single quote (') instead of double quote (\") for resolution.\n"
                         "Example: ++name='${exp_name}'")

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

    def mount_check(v, cluster_cfg):
        if v.startswith(os.path.sep):
            logging.info(f"Checking if {v} is a mounted path")
            run_utils.check_if_mounted(cluster_cfg, v)

        elif "ais://" in v and ais_endpoint is not None:  # if the value is a string, check if its an ais path
            # Try to import ais module
            try:
                from aistore.sdk import Client

                ais_client = Client(ais_endpoint)


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
                            mount_check(item, cluster_cfg)

                elif isinstance(v, str):  # if the value is a string, check if its an absolute a path
                    mount_check(v, cluster_cfg)

    check_mounted_path(script_config, cluster_config)

def update_exp_manager_runtime(script_config, cluster_cfg):
    if 'max_runtime' in cluster_cfg:
        with open_dict(script_config):
            if 'exp_manager' not in script_config:
                raise ValueError("exp_manager config not found in the script config file")

            script_config['exp_manager']['max_time_per_run'] = cluster_cfg['max_runtime']
            logging.info(f"Setting exp_manager.max_time_per_run to {cluster_cfg['max_runtime']}")


def get_execution_script(cluster_script_path, config_name, merged_cfg, cluster_cfg):
    # Create the command to run the script
    cmd = """
nvidia-smi && \
export PYTHONPATH=$PYTHONPATH:/nemo_run/code && \
export HF_TOKEN={HF_TOKEN} && \
export WANDB_API_KEY={WANDB} && \
find /results/ -name '*-unfinished' -type f -delete && \ 
cd {cluster_script_dir} && \
python -u -B {cluster_script_path} --config-path "/results/configs" --config-name "{config_name}" && \
cd /results && \
ls -l;
"""
    wandb_key = os.environ.get("WANDB", os.environ.get("WANDB_API_KEY", os.environ.get("WANDB_KEY", "")))

    format_dict = dict(
        cluster_script_dir=os.path.dirname(cluster_script_path),
        cluster_script_path=os.path.basename(cluster_script_path),
        config_name=config_name,
        HF_TOKEN=os.getenv('HF_TOKEN', ''),
        WANDB=wandb_key,
    )

    cmd = cmd.format(**format_dict)
    return cmd


@hydra_runner(config_path='conf', config_name='run_local')
def main(cluster_cfg):
    script_path = cluster_cfg.script
    script_config_path = cluster_cfg.script_config
    results_dir = cluster_cfg.results_dir

    script_path = Path(script_path).absolute()
    script_config_path = Path(script_config_path).absolute()

    gather_mounts(cluster_cfg)

    # Add the results directory to the cluster config as a mount path
    run_utils.add_mount_path(results_dir, '/results', cluster_cfg)

    cluster_script_path = check_root_path(script_path, NEMO_ROOT)

    # Create results and logdir
    log_dir = cluster_cfg.get('log_dir', os.path.join(results_dir, 'logs'))
    run_utils.create_remote_directory([results_dir, log_dir], cluster_cfg)

    script_config = OmegaConf.load(script_config_path)
    merged_config = merge_configs(script_config, cluster_cfg)
    update_exp_manager_runtime(merged_config, cluster_cfg)

    # perform path checks
    check_config_mount_paths(merged_config, cluster_cfg)

    # Resolve experiment name
    exp_name = cluster_cfg.exp_name
    if exp_name is None:
        if 'exp_manager' in merged_config and 'name' in merged_config['exp_manager']:
            exp_name = merged_config['exp_manager']['name']
        else:
            raise ValueError(
                "Experiment name not provided in the run config file (`exp_name`)) or the cluster config (inside exp_manager.name)"
            )

    with run.Experiment(exp_name) as exp:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config_name = f"{exp_name}_{timestamp}_config.yaml"
        cmd = get_execution_script(cluster_script_path, config_name, merged_config, cluster_cfg)

        # Create the remote config file
        config_dir = os.path.join(results_dir, 'configs')
        run_utils.create_remote_config(merged_config, config_name, config_dir, cluster_cfg)

        job_name = f"{exp_name}_job"
        num_gpus = cluster_cfg.get('num_gpus', merged_config['trainer']['devices'])
        if isinstance(num_gpus, list):
            num_gpus = len(num_gpus)
        if num_gpus == -1:
            num_gpus = 1 if cluster_cfg['executor'] == 'local' else 8
            logging.warning(f"\n\nSetting num_gpus to {num_gpus} as it was set to -1\n\n")

        num_nodes = cluster_cfg.get('num_nodes', merged_config['trainer'].get('num_nodes', 1))
        cluster_cfg = OmegaConf.to_object(cluster_cfg)

        run_utils.add_task(
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
            run_after=cluster_cfg.get('run_after', None),
        )

        run_utils.run_exp(exp, cluster_cfg)


if __name__ == '__main__':
    main()
