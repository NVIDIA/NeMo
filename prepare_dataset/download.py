import os
import hydra

from utils import download_single_file


@hydra.main(config_path="../conf", config_name="config")
def main(cfg) -> None:
    data_cfg = cfg["data_preparation"]
    PILE_URL_TRAIN = "https://the-eye.eu/public/AI/pile/train/"
    data_save_dir = data_cfg["data_save_dir"]
    file_number = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    url = f"{PILE_URL_TRAIN}{file_number:02d}.jsonl.zst"
    downloaded_path = download_single_file(url, data_save_dir, f"{file_number:02d}.jsonl.zst")

if __name__ == "__main__":
    main()
