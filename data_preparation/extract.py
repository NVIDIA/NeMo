import os
import multiprocessing

import hydra


@hydra.main(config_path="../conf", config_name="config")
def main(cfg) -> None:
    """Function to extract the pile dataset files on BCM.
    
    Arguments:
        cfg: main config file.
    """
    import utils

    bignlp_path = cfg.bignlp_path
    data_cfg = cfg.data_preparation
    data_dir = cfg.data_dir
    assert data_dir is not None, "data_dir must be a valid path."

    file_number = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    downloaded_path = os.path.join(data_dir, f"{file_number:02d}.jsonl.zst")
    output_file = f"{file_number:02d}.jsonl"
    utils.extract_single_zst_file(downloaded_path, data_dir, output_file)
    os.remove(downloaded_path)


def extract_bcp(cfg, file_numbers) -> None:
    """Function to extract the pile dataset files on BCP.
    
    Arguments:
        cfg: main config file.
        file_numbers: list of file numbers to extract.
    """
    from . import utils

    bignlp_path = cfg.bignlp_path
    data_cfg = cfg.data_preparation
    data_dir = cfg.data_dir
    assert data_dir is not None, "data_dir must be a valid path."

    proc_list = []
    for file_number in file_numbers:
        downloaded_path = os.path.join(data_dir, f"{file_number:02d}.jsonl.zst")
        output_file = f"{file_number:02d}.jsonl"
        print(f"Extracting file {downloaded_path}")
        p = multiprocessing.Process(target=utils.extract_single_zst_file, args=(downloaded_path, data_dir, output_file))
        proc_list.append(p)
        p.start()

    for proc in proc_list:
        proc.join()

    for file_number in file_numbers:
        downloaded_path = os.path.join(data_dir, f"{file_number:02d}.jsonl.zst")
        os.remove(downloaded_path)


if __name__ == "__main__":
    main()
