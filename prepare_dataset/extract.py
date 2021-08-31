import os

import hydra

from utils import extract_single_zst_file


@hydra.main(config_path="../conf", config_name="config")
def main(cfg) -> None:
    data_cfg = cfg["data_preparation"]
    data_save_dir = data_cfg["data_save_dir"]
    file_number = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    downloaded_path = os.path.join(data_save_dir, f"{file_number:02d}.jsonl.zst")
    output_file = f"{file_number:02d}.jsonl"
    extract_single_zst_file(downloaded_path, data_save_dir, output_file)
    os.remove(downloaded_path)


if __name__ == "__main__":
    main()
