import os
import multiprocessing

import hydra
import utils


@hydra.main(config_path="../../../conf", config_name="config")
def main(cfg) -> None:
    """Function to extract the pile dataset files on BCM.

    Arguments:
        cfg: main config file.
    """
    bignlp_path = cfg.get("bignlp_path")
    data_cfg = cfg.get("data_preparation")
    data_dir = cfg.get("data_dir")
    rm_downloaded = data_cfg.get("rm_downloaded")
    assert data_dir is not None, "data_dir must be a valid path."

    if cfg.get("cluster_type") == "bcm":
        file_number = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
        downloaded_path = os.path.join(data_dir, f"{file_number:02d}.jsonl.zst")
        output_file = f"{file_number:02d}.jsonl"
        utils.extract_single_zst_file(downloaded_path, data_dir, output_file, rm_downloaded)
    elif cfg.get("cluster_type") == "bcp":
        file_numbers = data_cfg.get("file_numbers")
        # Downloading the files
        files_list = utils.convert_file_numbers(file_numbers)
        # Assumes launched via mpirun:
        #   mpirun -N <nnodes> -npernode 1 ...
        wrank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
        wsize = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
        files_list_groups = utils.split_list(files_list, wsize)
        files_to_extract = files_list_groups[wrank]
        proc_list = []
        for file_number in files_to_extract:
            downloaded_path = os.path.join(data_dir, f"{file_number:02d}.jsonl.zst")
            output_file = f"{file_number:02d}.jsonl"
            # TODO: Consider multiprocessing.Pool instead.
            proc = multiprocessing.Process(
                target=utils.extract_single_zst_file,
                args=(downloaded_path, data_dir, output_file, rm_downloaded),
            )
            proc_list.append(proc)
            proc.start()

        for proc in proc_list:
            proc.join()


if __name__ == "__main__":
    main()
