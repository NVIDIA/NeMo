import os

import hydra

import utils


@hydra.main(config_path="../conf", config_name="config")
def main(cfg) -> None:
    bignlp_path = cfg["bignlp_path"]
    data_cfg = cfg["data_preparation"]
    data_save_dir = data_cfg["data_save_dir"]
    assert data_save_dir is not None, "data_save_dir must be a valid path."
    full_data_save_dir = os.path.join(bignlp_path, data_save_dir)

    file_number = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    downloaded_path = os.path.join(full_data_save_dir, f"{file_number:02d}.jsonl.zst")
    output_file = f"{file_number:02d}.jsonl"
    utils.extract_single_zst_file(downloaded_path, full_data_save_dir, output_file)
    os.remove(downloaded_path)
    

if __name__ == "__main__":
    main()
