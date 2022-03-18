import os
import multiprocessing

import hydra
import utils


@hydra.main(config_path="../../../conf", config_name="config")
def main(cfg):
    """Function to download the pile dataset files on BCM.
    
    Arguments:
        cfg: main config file.
    """
    bignlp_path = cfg.get("bignlp_path")
    data_cfg = cfg.get("data_preparation")
    data_dir = cfg.get("data_dir")
    pile_url_train = data_cfg.get("the_pile_url")
    assert data_dir is not None, "data_dir must be a valid path."

    if cfg.get("cluster_type") == "bcm":
        file_number = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
        url = f"{pile_url_train}{file_number:02d}.jsonl.zst"
        output_file = f"{file_number:02d}.jsonl.zst"
        downloaded_path = utils.download_single_file(url, data_dir, output_file)
    if cfg.get("cluster_type") == "bcp":
        file_numbers = data_cfg["file_numbers"]
        # Downloading the files
        files_list = utils.convert_file_numbers(file_numbers)
        # Assumes launched via mpirun:
        #   mpirun -N <nnodes> -npernode <preproc_npernode> ...
        # where preproc_npernode is set in dataprep config -> bcp config
        wrank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
        wsize = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
        files_list_groups = utils.split_list(files_list, wsize)
        files_to_download = files_list_groups[wrank]
        proc_list = []
        for file_number in files_to_download:
            url = f"{pile_url_train}{file_number:02d}.jsonl.zst"
            output_file = f"{file_number:02d}.jsonl.zst"
            # TODO: Consider multiprocessing.Pool instead.
            proc = multiprocessing.Process(
                target=utils.download_single_file,
                args=(url, data_dir, output_file))
            proc_list.append(proc)
            proc.start()

        for proc in proc_list:
            proc.join()


if __name__ == "__main__":
    main()
