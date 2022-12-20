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

"""
Multi-worker mC4 data downloading.
Example usage:
 python download.py \
    --c4-path=<path/to/c4> \
    --git-lfs-path=<path/to/git/lfs/folder> \
    --worker-mapping-file=<path/to/download_mapping_file>
"""

import argparse
import os
import sys
import time

from prepare import LANG_SPLIT, setup_git_lfs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download (m)C4")
    parser.add_argument("--c4-path", help="Path to (m)C4 dataset repo folder", required=True)
    parser.add_argument("--git-lfs-path", help="Path to git lfs", required=True)
    parser.add_argument("--worker-mapping-file", help="Decide which worker download which languages", required=True)
    parser.add_argument("--bcp", action="store_true", help="Whether on BCP platform")
    args = parser.parse_args()

    setup_git_lfs(args.git_lfs_path)
    if args.bcp:
        task_id = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))  # assume exec with mpirun
    else:  # on slurm based platforms
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    with open(args.worker_mapping_file) as f:
        mapping = f.readlines()
    languages = mapping[task_id].strip().split(",")
    print(" ****** Task ID {:02d} is preparing to download {:}...".format(task_id, languages))

    lang_split_dict = {}
    for lang in LANG_SPLIT:
        splits = LANG_SPLIT[lang]
        for split, pattern in splits:
            lang_split_dict[split] = pattern

    c4_path = args.c4_path
    start_time = time.time()
    for lang in languages:
        print(" ****** Task ID {:02d} starts to download {:}...".format(task_id, lang))
        if lang in lang_split_dict:
            os.system(
                f"cd {c4_path} && " f"git -c lfs.concurrenttransfers=20 lfs pull --include '{lang_split_dict[lang]}'"
            )
        else:
            os.system(
                f"cd {c4_path} && "
                f"git -c lfs.concurrenttransfers=20 lfs pull --include 'multilingual/c4-{lang}.*.json.gz'"
            )
        print(" ****** Task ID {:02d} finished downloading {:}...".format(task_id, lang))
        print(" ****** Task ID {:02d} time elapsed {:.2f} min.".format(task_id, (time.time() - start_time) / 60))
