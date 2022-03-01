import os
import shutil
import subprocess
import time
import glob
import argparse

"""
Example usage:
 python preprocess.py \
    --worker-mapping-file=<path/to/preprocess_mapping_file> \
    --output-path=<output/path> \
    --tokenizer-library <some_tokenizer_lib> \
    --tokenizer-model <some_tokenizer_model> \
    --dataset-impl mmap \
    --workers 80  \
    --preproc-folder \
    --apply-ftfy 
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess (m)C4", allow_abbrev=False)
    parser.add_argument("--rm-downloaded", help="Whether to remove original downloaded data", action="store_true")
    parser.add_argument("--output-path", help="Path to store output bin files", required=True)
    parser.add_argument("--worker-mapping-file", help="Decide which worker download which languages", required=True)
    parser.add_argument("--workers-per-node", default=int(os.environ.get("SLURM_NTASKS_PER_NODE", 1)),
                        help="Number of workers per node in preprocessing step", type=int)
    args, other_args = parser.parse_known_args()

    # TODO might be some issues on bcp
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = args.workers_per_node

    with open(args.worker_mapping_file) as f:
        mapping = f.readlines()
    lang_splits = []
    if task_id * world_size + rank < len(mapping):
        lang_splits = mapping[task_id * world_size + rank].strip().split(',')
    print(" ****** Task ID {:02d} Rank {:02d} is preparing to preprocess {:}...".format(task_id, rank, lang_splits))

    # TODO: delete later; after ftfy installed in docker
    if rank == 0:
        os.system("pip install ftfy")
    else:
        time.sleep(10)

    os.makedirs(args.output_path, exist_ok=True)
    start_time = time.time()
    cmd = ["python", "/opt/bignlp/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py"]
    for split in lang_splits:
        print(" ****** Task ID {:02d} Rank {:02d} starts to preprocess {:}...".format(
            task_id, rank, os.path.basename(split)))
        input_arg = ["--input", split]
        output_arg = ["--output-prefix", os.path.join(args.output_path, os.path.basename(split))]
        subprocess.check_call(cmd + input_arg + output_arg + other_args)
        print(" ****** Task ID {:02d} Rank {:02d} finished preprocessing {:}...".format(
            task_id, rank, os.path.basename(split)))
        print(" ****** Task ID {:02d} Rank {:02d} time elapsed {:.2f} min.".format(
            task_id, rank, (time.time() - start_time) / 60))
        if args.rm_downloaded:
            for f in os.listdir(split):
                os.remove(os.readlink(os.path.join(split, f)))
            shutil.rmtree(split)

