"""Module to launch training jobs using bignlp-scripts."""

import os
import shutil
import subprocess

from omegaconf import OmegaConf


def run_training(file_name: str, model_name: str, results_dir: str, cfg: OmegaConf) -> int:
    """
    Launch a training job for the given model name and config file, using bignlp-scripts.

    :param str file_name: name of the file configuration to be selected for training with bignlp-scripts.
    :param str model_name: model type to be run, usually gpt3, t5 or mt5.
    :param str results_dir: path to the directory where the results will be stored.
    :param OmegaConf cfg: OmegaConf object with full configuration for the HP tool.
    :return: SLURM job_id of the training job that was launched.
    :rtype: int
    """
    # Copy cluster config to bignlp-scripts.
    bignlp_scripts_path = cfg.get("bignlp_scripts_path")
    cluster_cfg = cfg.get("cluster")
    dst = os.path.join(bignlp_scripts_path, "conf/cluster/bcm.yaml")
    copy_cluster_config()
    print(f"Copied cluster config to {dst}")

    # Generate string of hydra overrides for bignlp-scripts.
    overrides_str = generate_overrides_str(cfg)

    bignlp_ci = f"BIGNLP_CI=1" if bool(os.getenv("BIGNLP_CI")) else ""
    main_path = os.path.join(bignlp_scripts_path, "main.py")
    cmd = f"HYDRA_FULL_ERROR=1 {bignlp_ci} python3 {main_path} {overrides_str} "

    # Launch job with command cmd.
    job_output = subprocess.check_output([cmd], shell=True).decode("utf-8")
    job_id = job_output.split(" ")[-1]
    print(f"Submitted Training script with job id: {job_id}")
    return job_id


def copy_config(cfg: OmegaConf, dst: str) -> None:
    """
    Copies OmegaConf configuration to a dst file.

    :param OmegaConf cfg: OmegaConfg object with the config to be stored in a file.
    :param str dst: destination path to where the config will be stored. Must be a yaml file.
    :return: None
    """
    with open(dst, "w") as f:
        OmegaConf.save(config=cfg, f=f)


def generate_overrides_str(
    file_name: str, model_name: str, results_dir: str, cfg: OmegaConf
) -> str:
    """
    Generates string with hydra-like parameter overrides for bignlp-scripts.

    :param str file_name: name of the file configuration to be selected for training with bignlp-scripts.
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
    bignlp_scripts_path = cfg.get("bignlp_scripts_path")
    data_dir = cfg.get("data_dir")
    container_mounts = cfg.get("container_mounts")
    api_key_file = cfg.get("wandb").get("api_key_file")

    overrides_str = (
        f"training={training_model} "
        f"stages=[training] "
        f"cluster_type={cluster_type} "
        f"base_results_dir={results_dir} "
        f"\"container='{container}'\" "
        f"bignlp_path={bignlp_scripts_path} "
        f"data_dir={data_dir} "
        f"training.exp_manager.create_checkpoint_callback=False "
        f"container_mounts={container_mounts} "
        f"wandb_api_key_file={api_key_file} "
    )
    return overrides_str
