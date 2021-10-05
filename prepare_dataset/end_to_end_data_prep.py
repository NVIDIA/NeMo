import sys
import os
import subprocess

import hydra

import utils


def create_slurm_file(
    new_file_name,
    code_file_name,
    log_dir="./",
    flags="",
    depend=None,
    time="04:00:00",
    exclusive=True,
    file_numbers="0",
    nodes=1,
    partition="A100",
):
    path_to_file = os.path.join(os.environ.get("PWD"), new_file_name)
    path_to_code = os.path.join(os.environ.get("PWD"), code_file_name)
    task = code_file_name.split("/")[-1].split(".")[0]
    with open(path_to_file, "w") as f:
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
        f.writelines("cd $SLURM_SUBMIT_DIR\n")
        f.writelines(f"srun {flags} python3 {path_to_code} &\n")
        f.writelines("wait\n")
    return path_to_file


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    # Read config
    data_cfg = cfg["data_preparation"]
    bignlp_path = cfg.get("bignlp_path")
    slurm_cfg = data_cfg["slurm"]

    # Data preparation config
    download_the_pile = data_cfg.get("download_the_pile")
    file_numbers = data_cfg["file_numbers"]
    data_save_dir = data_cfg["data_save_dir"]
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
        utils.download_single_file(url=download_vocab_url, save_dir=vocab_save_dir, file_name="vocab.txt")

    # Download merges
    if download_merges_url is not None:
        assert merges_save_dir is not None, "merges_save_dir must be a valid path."
        utils.download_single_file(url=download_merges_url, save_dir=merges_save_dir, file_name="merges.txt")

    dependency = None
    assert isinstance(download_the_pile, bool), "download_the_pile must be bool."
    if download_the_pile:
        # Download The Pile dataset files
        download_file_name = "download_script.sh"
        path_to_download_file = create_slurm_file(
            new_file_name=download_file_name,
            code_file_name="download.py",
            log_dir=log_dir,
            time=time_limit,
            file_numbers=file_numbers,
            nodes=nodes,
            partition=partition,
        )
        job_id_1 = subprocess.check_output(
            [f"sbatch --parsable {path_to_download_file}"], shell=True
        )
        job_id_1 = job_id_1.decode("utf-8")
        print(f"Submitted Download script with job id: {job_id_1}")
        dependency = f"aftercorr:{job_id_1}"

        # Extract The Pile dataset files
        extract_file_name = "extract_script.sh"
        path_to_extract_file = create_slurm_file(
            new_file_name=extract_file_name,
            code_file_name="extract.py",
            log_dir=log_dir,
            depend=dependency,
            time=time_limit,
            file_numbers=file_numbers,
            nodes=nodes,
            partition=partition,
        )
        job_id_2 = subprocess.check_output(
            [f"sbatch --parsable {path_to_extract_file}"], shell=True
        )
        job_id_2 = job_id_2.decode("utf-8")
        print(f"Submitted Extract script with job id: {job_id_2}")
        dependency = f"aftercorr:{job_id_2}"








    assert isinstance(preprocess_data, bool), "preprocess_data must be bool."
    if preprocess_data:
        # Preprocess the dataset
        preprocess_file_name = "preprocess_script.sh"
        flags = (
            f"--container-image nvcr.io/nvidian/nemo_megatron-gpt:pyt21.09 "
            f"--container-mounts {bignlp_path}:/workspace/bignlp-scripts"
        )
        preprocess_code = f"/workspace/bignlp-scripts/prepare_dataset/preprocess.py"
        path_to_preprocess_file = create_slurm_file(
            new_file_name=preprocess_file_name,
            code_file_name=preprocess_code,
            log_dir=log_dir,
            flags=flags,
            depend=dependency,
            time=time_limit,
            file_numbers=file_numbers,
            nodes=nodes,
            partition=partition,
        )
        job_id_3 = subprocess.check_output(
            [f"sbatch --parsable {path_to_preprocess_file}"], shell=True
        )
        job_id_3 = job_id_3.decode("utf-8")
        print(f"Submitted Preprocessing script with job id: {job_id_3}")


if __name__ == "__main__":
    main()
