import os

import hydra

import utils


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    data_cfg = cfg["data_preparation"]
    PILE_URL_TRAIN = "https://the-eye.eu/public/AI/pile/train/"
    file_number = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    url = f"{PILE_URL_TRAIN}{file_number:02d}.jsonl.zst"
    data_save_dir = data_cfg["data_save_dir"]
    output_file = f"{file_number:02d}.jsonl.zst"
    downloaded_path = utils.download_single_file(url, data_save_dir, output_file)


if __name__ == "__main__":
    main()
