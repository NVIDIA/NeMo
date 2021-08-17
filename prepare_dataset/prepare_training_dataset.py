import sys
import os

import requests
import tqdm
import hydra
import submitit
import zstandard as zstd



def download_single_file(url, save_dir, file_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name)
    if os.path.exists(save_path):
        return save_path

    with requests.get(url, stream=True) as read_file, open(save_path, 'wb') as write_file:
        total_length = int(read_file.headers.get("content-length"))
        with tqdm.tqdm(
            total=total_length,
            unit="B",
            unit_scale=True,
            desc=file_name,
        ) as pbar:
            for chunk in read_file.iter_content(chunk_size=8192):
                if chunk:
                    write_file.write(chunk)
                    pbar.update(len(chunk))
    return save_path

def extract_single_zst_file(input_path, save_dir, file_name):
    total_length = os.stat(input_path).st_size
    with tqdm.tqdm(
        total=total_length,
        unit="B",
        unit_scale=True,
        desc=file_name,
    ) as pbar:
        dctx = zstd.ZstdDecompressor()
        read_size = 131075
        write_size = int(read_size * 4)
        save_path = os.path.join(save_dir, file_name)
        with open(input_path, 'rb') as in_f, open(save_path, 'wb') as out_f:
            for chunk in dctx.read_to_iter(in_f, read_size=read_size, write_size=write_size):
                out_f.write(chunk)
                pbar.update(read_size)

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

def download_and_extract_the_pile(cfg):
    #job_file = os.path.join(os.environ.get("PWD"), 'script_download_all_files.sh')
    PILE_URL_TRAIN = "https://the-eye.eu/public/AI/pile/train/"
    data_save_dir = cfg["data_save_dir"]
    file_numbers = _process_number_range(cfg["file_numbers"])
    for file_num in file_numbers:
        url = f"{PILE_URL_TRAIN}{file_num:02d}.jsonl.zst"
        downloaded_path = download_single_file(url, data_save_dir, f"{file_num:02d}.jsonl.zst")
        extract_single_zst_file(downloaded_path, data_save_dir, f"{file_num:02d}.jsonl")

def create_download_file():
    with open(download_file) as f:
        f.writelines("#!/bin/bash\n")
        f.writelines("#SBATCH --nodes=1\n")
        f.writelines("#SBATCH --job-name=bignlp:download_all_pile_files\n")
        f.writelines("#SBATCH --requeue\n")
        f.writelines("#SBATCH --exclusive\n")
        f.writelines("#SBATCH --time=04:00:00\n")
        f.writelines("#SBATCH --array=0-29\n")
        f.writelines("#SBATCH -o log-download-%j_%a.out\n")
        f.writelines("cd $SLURM_SUBMIT_DIR")
        f.writelines("srun python download.py &")
        f.writelines("wait")


@hydra.main(config_path="../conf", config_name="config")
def main(cfg) -> None:
    data_cfg = cfg["data_preparation"]

    # Download vocab
    if data_cfg.get("download_vocab_url") is not None:
        download_vocab(cfg=data_cfg)

    # Download merges
    if data_cfg.get("download_vocab_url") is not None:
        download_merges(cfg=data_cfg)

    

    # Download dataset files
    if data_cfg.get("download_the_pile") is not None:
        download_and_extract_the_pile(cfg=data_cfg)


if __name__ == "__main__":
    main()
