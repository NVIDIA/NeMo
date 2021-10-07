import os

import hydra

import utils


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    bignlp_path = cfg["bignlp_path"]
    data_cfg = cfg["data_preparation"]
    data_save_dir = data_cfg["data_save_dir"]
    assert data_save_dir is not None, "data_save_dir must be a valid path."
    full_data_save_dir = os.path.join(bignlp_path, data_save_dir)

    PILE_URL_TRAIN = "https://the-eye.eu/public/AI/pile/train/"
    file_number = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    url = f"{PILE_URL_TRAIN}{file_number:02d}.jsonl.zst"
    output_file = f"{file_number:02d}.jsonl.zst"
    downloaded_path = utils.download_single_file(url, full_data_save_dir, output_file)


if __name__ == "__main__":
    main()
