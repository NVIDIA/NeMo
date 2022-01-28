import os
import multiprocessing

import hydra


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    """Function to download the pile dataset files on BCM.
    
    Arguments:
        cfg: main config file.
    """
    import utils

    bignlp_path = cfg.bignlp_path
    data_cfg = cfg.data_preparation
    data_dir = cfg.data_dir
    pile_url_train = data_cfg.the_pile_url
    assert data_dir is not None, "data_dir must be a valid path."

    file_number = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    url = f"{pile_url_train}{file_number:02d}.jsonl.zst"
    output_file = f"{file_number:02d}.jsonl.zst"
    downloaded_path = utils.download_single_file(url, data_dir, output_file)


def download_bcp(cfg, file_numbers):
    """Function to download the pile dataset files on BCP.
    
    Arguments:
        cfg: main config file.
        file_numbers: list of file numbers to download.
    """
    from . import utils

    bignlp_path = cfg.bignlp_path
    data_cfg = cfg.data_preparation
    data_dir = cfg.data_dir
    pile_url = data_cfg.the_pile_url
    assert data_dir is not None, "data_dir must be a valid path."

    proc_list = []
    for file_number in file_numbers:
        url = f"{pile_url}/{file_number:02d}.jsonl.zst"
        output_file = f"{file_number:02d}.jsonl.zst"
        print(f"Downloading file from {url}")
        p = multiprocessing.Process(target=utils.download_single_file, args=(url, data_dir, output_file))
        proc_list.append(p)
        p.start()

    for proc in proc_list:
        proc.join()


if __name__ == "__main__":
    main()
