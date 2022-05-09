import os
import subprocess
from time import sleep

import psutil
import hydra
import utils


@hydra.main(config_path="../../../conf", config_name="config")
def main(cfg):
    bignlp_path = cfg.get("bignlp_path")
    data_config = cfg.get("data_config")
    data_cfg = cfg.get("data_preparation")
    data_dir = cfg.get("data_dir")
    rm_extracted = data_cfg.get("rm_extracted")
    tokenizer_type = data_cfg.get("tokenizer_type")
    assert data_dir is not None, "data_dir must be a valid path"

    # Vocab
    vocab_dir = data_cfg.get("vocab_save_dir")
    assert vocab_dir is not None, "vocab_save_dir must be a valid path."
    if "gpt" in tokenizer_type.lower():
        vocab_path = os.path.join(bignlp_path, vocab_dir, "vocab.json")
    else:
        vocab_path = os.path.join(bignlp_path, vocab_dir, "vocab.txt")

    # Merges
    merges_dir = data_cfg.get("merges_save_dir")
    assert merges_dir is not None, "merges_save_dir must be a valid path."
    merges_path = os.path.join(bignlp_path, merges_dir, "merges.txt")

    # This compile doesn't seem to do anything. It compiles
    # "helpers.cpython-38-x86_64-linux-gnu.so", but since that file already
    # exists, it doesn't do anything. Force make via: touch helpers.cpp
    megatron_dir = "/opt/bignlp/NeMo/nemo/collections/nlp/data/language_modeling/megatron"
    compiled_helpers_lib = os.path.join(megatron_dir, "compiled_helpers_lib")
    compilecmd = (
        f"cd /opt/bignlp/NeMo; git rev-parse HEAD; "
        f"cd {megatron_dir}; "
        f"touch helpers.cpp; make;"
    )

    code_path = "/opt/bignlp/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py"
    runcmd = (
        f"cd {megatron_dir}; "
        f'export PYTHONPATH="/opt/bignlp/NeMo/.:$PYTHONPATH"; '
        f'export TRANSFORMERS_CACHE="/temp_root/.cache/"; '
        f"CUDA_VISIBLE_DEVICES=0,4,2,6,1,5,3,7 python3 {code_path} " + "{flags}"
    )

    if cfg.get("cluster_type") == "bcm":
        file_number = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
        extracted_path = os.path.join(data_dir, f"{file_number:02d}.jsonl")
        # TODO: find better way to do this
        output_prefix = os.path.join(
            data_dir, f"my-{'t5' if 't5' in data_config else 'gpt3'}_{file_number:02d}"
        )

        flags = (
            f"--input {extracted_path} "
            f"--output-prefix {output_prefix} "
            f"--vocab {vocab_path} "
            f"--merge-file {merges_path} "
            f"--dataset-impl mmap "
            f"--tokenizer-library megatron "
            f"--tokenizer-type {tokenizer_type} "
            f"--workers $SLURM_CPUS_ON_NODE "
            f"--append-eod "
        )
        os.system(compilecmd)
        os.system(runcmd.format(flags=flags))
        if rm_extracted:
            os.remove(extracted_path)
    elif cfg.get("cluster_type") == "bcp":
        file_numbers = data_cfg.get("file_numbers")
        files_list = utils.convert_file_numbers(file_numbers)
        # Assumes launched via mpirun:
        #   mpirun -N <nnodes> -npernode 1 ...
        wrank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))
        wsize = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 0))
        lrank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))

        if lrank == 0:
            # Compile once per node. Should be one container instance per node.
            os.system(compilecmd)
            os.system(f"touch {compiled_helpers_lib}")
        else:
            while not os.path.exists(compiled_helpers_lib):
                sleep(1)

        files_list_groups = utils.split_list(files_list, wsize)
        files_to_preproc = files_list_groups[wrank]
        ncpus = psutil.cpu_count(logical=False)
        for file_number in files_to_preproc:
            extracted_path = os.path.join(data_dir, f"{file_number:02d}.jsonl")
            output_prefix = os.path.join(
                data_dir, f"my-{'t5' if 't5' in data_config else 'gpt3'}_{file_number:02d}"
            )

            flags = (
                f"--input {extracted_path} "
                f"--output-prefix {output_prefix} "
                f"--vocab {vocab_path} "
                f"--merge-file {merges_path} "
                f"--dataset-impl mmap "
                f"--tokenizer-library megatron "
                f"--tokenizer-type {tokenizer_type} "
                f"--workers {ncpus} "
                f"--append-eod "
            )
            proc = subprocess.Popen(runcmd.format(flags=flags), shell=True)
            proc.wait()
            if rm_extracted:
                os.remove(extracted_path)


if __name__ == "__main__":
    main()
