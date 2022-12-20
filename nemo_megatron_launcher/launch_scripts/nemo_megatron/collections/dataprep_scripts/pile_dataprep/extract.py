# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
import os

import hydra
import nemo_megatron.utils.file_utils as utils


@hydra.main(config_path="conf", config_name="config")
def main(cfg) -> None:
    """Function to extract the pile dataset files on BCM.

    Arguments:
        cfg: main config file.
    """
    data_dir = cfg.get("data_dir")
    rm_downloaded = cfg.get("rm_downloaded")
    assert data_dir is not None, "data_dir must be a valid path."

    if cfg.get("cluster_type") == "bcm":
        file_number = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
        downloaded_path = os.path.join(data_dir, f"{file_number:02d}.jsonl.zst")
        output_file = f"{file_number:02d}.jsonl"
        utils.extract_single_zst_file(downloaded_path, data_dir, output_file, rm_downloaded)
    elif cfg.get("cluster_type") == "bcp":
        file_numbers = cfg.get("file_numbers")
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
                target=utils.extract_single_zst_file, args=(downloaded_path, data_dir, output_file, rm_downloaded),
            )
            proc_list.append(proc)
            proc.start()

        for proc in proc_list:
            proc.join()


if __name__ == "__main__":
    main()
