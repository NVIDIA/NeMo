import sys
import os
import subprocess

import hydra
import omegaconf

from . import utils


def create_slurm_file(
    new_script_path,
    code_path,
    log_dir="./",
    flags="",
    hydra_args="",
    dependency=None,
    time="04:00:00",
    exclusive=True,
    requeue=True,
    file_numbers="0",
    nodes=1,
    partition="batch",
):
    task = code_path.split("/")[-1].split(".")[0]
    with open(new_script_path, "w") as f:
        f.writelines("#!/bin/bash\n")
        f.writelines("#SBATCH --nodes=1\n")
        if dependency is not None:
            f.writelines(f"#SBATCH --dependency=aftercorr:{dependency}\n")
        f.writelines(f"#SBATCH -p {partition}\n")
        f.writelines(f"#SBATCH --job-name=bignlp:{task}_all_pile_files\n")
        if requeue:
            f.writelines("#SBATCH --requeue\n")
        if exclusive:
            f.writelines("#SBATCH --exclusive\n")
        f.writelines(f"#SBATCH --time={time}\n")
        f.writelines(f"#SBATCH --array={file_numbers}%{nodes}\n")
        f.writelines(f"#SBATCH -o {log_dir}/log-{task}-%j_%a.out\n")
        f.writelines(f"srun {flags} python3 {code_path} {hydra_args} &\n")
        f.writelines("wait\n")


def run_data_preparation(cfg, hydra_args="", dependency=None):
    # Read config
    data_cfg = cfg["data_preparation"]
    bignlp_path = cfg.get("bignlp_path")
    container_mounts = cfg.get("container_mounts")
    slurm_cfg = data_cfg["slurm"]
    container = cfg["container"]

    # Data preparation config
    download_the_pile = data_cfg.get("download_the_pile")
    file_numbers = data_cfg["file_numbers"]
    preprocess_data = data_cfg.get("preprocess_data")
    download_vocab_url = data_cfg.get("download_vocab_url")
    download_merges_url = data_cfg.get("download_merges_url")
    vocab_save_dir = data_cfg.get("vocab_save_dir")
    merges_save_dir = data_cfg.get("merges_save_dir")
    log_dir = data_cfg.get("log_dir")

    # Slurm config
    partition = slurm_cfg["partition"]
    time_limit = slurm_cfg["time_limit"]
    nodes = slurm_cfg["nodes"]

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Download vocab
    if download_vocab_url is not None:
        assert vocab_save_dir is not None, "vocab_save_dir must be a valid path."
        utils.download_single_file(
            url=download_vocab_url, save_dir=vocab_save_dir, file_name="vocab.json"
        )

    # Download merges
    if download_merges_url is not None:
        assert merges_save_dir is not None, "merges_save_dir must be a valid path."
        utils.download_single_file(
            url=download_merges_url,
            save_dir=merges_save_dir,
            file_name="merges.txt",
        )

    # Process container-mounts.
    mounts_str = f"{bignlp_path}:{bignlp_path}"
    if container_mounts is not None:
        assert isinstance(container_mounts, omegaconf.listconfig.ListConfig), "container_mounts must be a list."
        for mount in container_mounts:
            if mount is not None and isinstance(mount, str):
                mounts_str += f",{mount}:{mount}"


    assert isinstance(download_the_pile, bool), "download_the_pile must be bool."
    if download_the_pile:
        # Download The Pile dataset files
        flags = (
            f"--container-image {container} "
            f"--container-mounts {mounts_str}"
        )
        download_script_path = os.path.join(
            bignlp_path, "data_preparation/download_script.sh"
        )
        download_code_path = os.path.join(bignlp_path, "data_preparation/download.py")
        create_slurm_file(
            new_script_path=download_script_path,
            code_path=download_code_path,
            log_dir=log_dir,
            flags=flags,
            hydra_args=hydra_args,
            dependency=dependency,
            time=time_limit,
            file_numbers=file_numbers,
            nodes=nodes,
            partition=partition,
        )
        job_id_1 = subprocess.check_output(
            [f"sbatch --parsable {download_script_path}"], shell=True
        )
        dependency = job_id_1.decode("utf-8")
        print(f"Submitted Download script with job id: {dependency}")

        # Extract The Pile dataset files
        flags = (
            f"--container-image {container} "
            f"--container-mounts {mounts_str}"
        )
        extract_script_path = os.path.join(
            bignlp_path, "data_preparation/extract_script.sh"
        )
        extract_code_path = os.path.join(bignlp_path, "data_preparation/extract.py")
        create_slurm_file(
            new_script_path=extract_script_path,
            code_path=extract_code_path,
            log_dir=log_dir,
            flags=flags,
            hydra_args=hydra_args,
            dependency=dependency,
            time=time_limit,
            file_numbers=file_numbers,
            nodes=nodes,
            partition=partition,
        )
        job_id_2 = subprocess.check_output(
            [f"sbatch --parsable {extract_script_path}"], shell=True
        )
        dependency = job_id_2.decode("utf-8")
        print(f"Submitted Extract script with job id: {dependency}")

    assert isinstance(preprocess_data, bool), "preprocess_data must be bool."
    if preprocess_data:
        # Preprocess the dataset
        flags = (
            f"--container-image {container} "
            f"--container-mounts {mounts_str}"
        )
        preprocess_script_path = os.path.join(
            bignlp_path, "data_preparation/preprocess_script.sh"
        )
        preprocess_code_path = os.path.join(
            bignlp_path, "data_preparation/preprocess.py"
        )
        create_slurm_file(
            new_script_path=preprocess_script_path,
            code_path=preprocess_code_path,
            log_dir=log_dir,
            flags=flags,
            hydra_args=hydra_args,
            dependency=dependency,
            time=time_limit,
            file_numbers=file_numbers,
            nodes=nodes,
            partition=partition,
        )
        job_id_3 = subprocess.check_output(
            [f"sbatch --parsable {preprocess_script_path}"], shell=True
        )
        dependency = job_id_3.decode("utf-8")
        print(f"Submitted Preprocessing script with job id: {dependency}")

    return dependency
