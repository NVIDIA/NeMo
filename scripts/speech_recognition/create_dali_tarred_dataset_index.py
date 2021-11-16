# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import copy
import json
import os
import glob
import shutil
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional
import subprocess

import joblib
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf, open_dict

"""
python create_dali_tarred_dataset_index.py \
    --tar_dir=<path to the directory which contains tarred dataset> \
    --workers=-1

"""

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(
    description="Convert an existing ASR dataset to tarballs compatible with TarredAudioToTextDataLayer."
)
parser.add_argument(
    "--tar_dir", default=None, type=str, required=True, help="Path to the existing dataset's manifest."
)

parser.add_argument('--workers', type=int, default=-1, help='Number of worker processes')
args = parser.parse_args()


def process_index_path(tar_paths, index_dir):
    index_paths = []
    for path in tar_paths:
        path = path.replace('.tar', '.index')
        base, name = os.path.split(path)
        index_path = os.path.join(index_dir, name)
        index_paths.append(index_path)

    return index_paths


def build_index(tarpath, indexfile):
    cmd = ['wds2idx', tarpath, indexfile]
    subprocess.run(cmd, capture_output=True)


def main(args):
    wds2idx_path = shutil.which('wds2idx')

    if wds2idx_path is None:
        logging.error("`wds2idx` is not installed. Please install NVIDIA DALI >= 1.7")
        exit(1)

    tar_files = list(glob.glob(os.path.join(args.tar_dir, "*.tar")))

    index_dir = os.path.join(args.tar_dir, "dali_index")
    if not os.path.exists(index_dir):
        os.makedirs(index_dir, exist_ok=True)

    index_paths = process_index_path(tar_files, index_dir)

    with joblib.Parallel(n_jobs=args.workers, verbose=len(tar_files)) as parallel:
        _ = parallel(delayed(build_index)(tarpath, indexfile) for tarpath, indexfile in zip(tar_files, index_paths))

    logging.info("Finished constructing index files !")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
