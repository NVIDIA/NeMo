import sys
import os

import requests
import tqdm
import hydra
import zstandard as zstd

from utils import download_single_file


def _process_number_range(number_str):
    final_list = []
    number_list = number_str.split(",")
    for num in number_list:
        try:
            num = int(num)
        except ValueError:
            start, end = num.split("-")
            final_list += range(int(start), int(end) + 1)
            continue
        final_list.append(num)

    return list(map(int, final_list))

def download_vocab(cfg):
    download_vocab_url = cfg["download_vocab_url"]
    vocab_save_dir = cfg.get("vocab_save_dir")
    assert vocab_save_dir is not None, "A directory must be given to store the vocab file."
    download_single_file(url=download_vocab_url, save_dir=vocab_save_dir, file_name="vocab.json")

def download_merges(cfg):
    download_merges_url = cfg["download_merges_url"]
    merges_save_dir = cfg.get("merges_save_dir")
    assert merges_save_dir is not None, "A directory must be given to store the merges file."
    download_single_file(url=download_merges_url, save_dir=merges_save_dir, file_name="merges.txt")

def create_slurm_file(new_file_name, code_file_name, flags="", time="04:00:00", exclusive=True, file_numbers="0", nodes=1, partition='A100'):
    path_to_file = os.path.join(os.environ.get("PWD"), new_file_name)
    path_to_code = os.path.join(os.environ.get("PWD"), code_file_name)
    task = code_file_name.split(".")[0]
    with open(path_to_file, 'w') as f:
        f.writelines("#!/bin/bash\n")
        f.writelines("#SBATCH --nodes=1\n")
        f.writelines(f"#SBATCH -p {partition}\n")
        f.writelines(f"#SBATCH --job-name=bignlp:{task}_all_pile_files\n")
        f.writelines("#SBATCH --requeue\n")
        if exclusive:
            f.writelines("#SBATCH --exclusive\n")
        f.writelines(f"#SBATCH --time={time}\n")
        f.writelines(f"#SBATCH --array={file_numbers}%{nodes}\n")
        f.writelines(f"#SBATCH -o log-{task}-%j_%a.out\n")
        f.writelines("cd $SLURM_SUBMIT_DIR\n")
        f.writelines(f"srun python {path_to_code} {flags} &\n")
        f.writelines("wait\n")
    return path_to_file


@hydra.main(config_path="../conf", config_name="config")
def main(cfg) -> None:
    print(cfg)
    data_cfg = cfg["data_preparation"]

    # Download vocab
    if data_cfg.get("download_vocab_url") is not None:
        download_vocab(cfg=data_cfg)

    # Download merges
    if data_cfg.get("download_vocab_url") is not None:
        download_merges(cfg=data_cfg)
    

    if data_cfg.get("download_the_pile") is not None:
        pass
        # Download dataset files
        #download_file_name = 'download_script.sh'
        #path_to_download_file = create_slurm_file(download_file_name, "download.py")
        #os.system(f"sbatch {path_to_download_file}")

        # Extract dataset files
        #extract_file_name = 'extract_script.sh'
        #path_to_extract_file = create_slurm_file(extract_file_name, "extract.py")
        #os.system(f"sbatch {path_to_extract_file}")

    preprocess_data = data_cfg.get("preprocess_data")
    pwd = os.environ.get("PWD")
    if preprocess_data is not None:
        assert isinstance(preprocess_data, bool), "The value of preprocess_data must be a bool."
        if preprocess_data:
            preprocess_file_name = "preprocess_script.sh"
            flags = f"--container-image nvcr.io#nvidia/pytorch:21.06-py3 \
                     --container-mounts {pwd}/..:/workspace/bignlp-scripts"
            path_to_preprocess_file = create_slurm_file(preprocess_file_name, "preprocess.py", flags)
            os.system(f"sbatch {path_to_preprocess_file}")

if __name__ == "__main__":
    main()
