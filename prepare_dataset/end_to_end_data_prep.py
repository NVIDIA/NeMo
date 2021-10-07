import sys
import os
import subprocess

import hydra

import utils


def create_slurm_file(
    new_script_path,
    code_path,
    log_dir="./",
    flags="",
    depend=None,
    time="04:00:00",
    exclusive=True,
    file_numbers="0",
    nodes=1,
    partition="A100",
):
    task = code_path.split("/")[-1].split(".")[0]
    with open(new_script_path, "w") as f:
        f.writelines("#!/bin/bash\n")
        f.writelines("#SBATCH --nodes=1\n")
        if depend is not None:
            f.writelines(f"#SBATCH --depend={depend}\n")
        f.writelines(f"#SBATCH -p {partition}\n")
        f.writelines(f"#SBATCH --job-name=bignlp:{task}_all_pile_files\n")
        f.writelines("#SBATCH --requeue\n")
        if exclusive:
            f.writelines("#SBATCH --exclusive\n")
        f.writelines(f"#SBATCH --time={time}\n")
        f.writelines(f"#SBATCH --array={file_numbers}%{nodes}\n")
        f.writelines(f"#SBATCH -o {log_dir}/log-{task}-%j_%a.out\n")
        f.writelines(f"srun {flags} python3 {code_path} &\n")
        f.writelines("wait\n")


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    # Read config
    data_cfg = cfg["data_preparation"]
    bignlp_path = cfg.get("bignlp_path")
    slurm_cfg = data_cfg["slurm"]
    container = cfg["container"]

    # Data preparation config
    download_the_pile = data_cfg.get("download_the_pile")
    file_numbers = data_cfg["file_numbers"]
    data_save_dir = data_cfg["data_save_dir"]  ################################################################
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

    full_log_dir = os.path.join(bignlp_path, log_dir)
    if not os.path.exists(full_log_dir):
        os.makedirs(full_log_dir)

    # Download vocab
    if download_vocab_url is not None:
        assert vocab_save_dir is not None, "vocab_save_dir must be a valid path."
        if bignlp_path not in vocab_save_dir:
            full_vocab_save_dir = os.path.join(bignlp_path, vocab_save_dir)
        utils.download_single_file(url=download_vocab_url, save_dir=full_vocab_save_dir, file_name="vocab.txt")

    # Download merges
    if download_merges_url is not None:
        assert merges_save_dir is not None, "merges_save_dir must be a valid path."
        if bignlp_path not in merges_save_dir:
            full_merges_save_dir = os.path.join(bignlp_path, merges_save_dir)
        utils.download_single_file(url=download_merges_url, save_dir=full_merges_save_dir, file_name="merges.txt")

    dependency = None
    assert isinstance(download_the_pile, bool), "download_the_pile must be bool."
    if download_the_pile:
        # Download The Pile dataset files
        flags = (
            f"--container-image {container} "
            f"--container-mounts {bignlp_path}:{bignlp_path}"
        )
        download_script_path = os.path.join(bignlp_path, "prepare_dataset/download_script.sh")
        download_code_path = os.path.join(bignlp_path, "prepare_dataset/download.py")
        create_slurm_file(
            new_script_path=download_script_path,
            code_path=download_code_path,
            log_dir=full_log_dir,
            flags=flags,
            time=time_limit,
            file_numbers=file_numbers,
            nodes=nodes,
            partition=partition,
        )
        job_id_1 = subprocess.check_output(
            [f"sbatch --parsable {download_script_path}"], shell=True
        )
        job_id_1 = job_id_1.decode("utf-8")
        print(f"Submitted Download script with job id: {job_id_1}")
        dependency = f"aftercorr:{job_id_1}"

        # Extract The Pile dataset files
        flags = (
            f"--container-image {container} "
            f"--container-mounts {bignlp_path}:{bignlp_path}"
        )
        extract_script_path = os.path.join(bignlp_path, "prepare_dataset/extract_script.sh")
        extract_code_path = os.path.join(bignlp_path, "prepare_dataset/extract.py")
        create_slurm_file(
            new_script_path=extract_script_path,
            code_path=extract_code_path,
            log_dir=full_log_dir,
            flags=flags,
            depend=dependency,
            time=time_limit,
            file_numbers=file_numbers,
            nodes=nodes,
            partition=partition,
        )
        job_id_2 = subprocess.check_output(
            [f"sbatch --parsable {extract_script_path}"], shell=True
        )
        job_id_2 = job_id_2.decode("utf-8")
        print(f"Submitted Extract script with job id: {job_id_2}")
        dependency = f"aftercorr:{job_id_2}"

    assert isinstance(preprocess_data, bool), "preprocess_data must be bool."
    if preprocess_data:
        # Preprocess the dataset
        flags = (
            f"--container-image {container} "
            f"--container-mounts {bignlp_path}:{bignlp_path}"
        )
        preprocess_script_path = os.path.join(bignlp_path, "prepare_dataset/preprocess_script.sh")
        preprocess_code_path = os.path.join(bignlp_path, "prepare_dataset/preprocess.py")
        create_slurm_file(
            new_script_path=preprocess_script_path,
            code_path=preprocess_code_path,
            log_dir=full_log_dir,
            flags=flags,
            depend=dependency,
            time=time_limit,
            file_numbers=file_numbers,
            nodes=nodes,
            partition=partition,
        )
        job_id_3 = subprocess.check_output(
            [f"sbatch --parsable {preprocess_script_path}"], shell=True
        )
        job_id_3 = job_id_3.decode("utf-8")
        print(f"Submitted Preprocessing script with job id: {job_id_3}")


if __name__ == "__main__":
    main()
