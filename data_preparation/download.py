import os

import hydra

import utils


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    bignlp_path = cfg.bignlp_path
    data_cfg = cfg.data_preparation
    data_dir = cfg.data_dir
    pile_url_train = data_cfg.the_pile_url
    assert data_dir is not None, "data_dir must be a valid path."

    file_number = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    url = f"{pile_url_train}{file_number:02d}.jsonl.zst"
    output_file = f"{file_number:02d}.jsonl.zst"
    downloaded_path = utils.download_single_file(url, data_dir, output_file)


if __name__ == "__main__":
    main()
