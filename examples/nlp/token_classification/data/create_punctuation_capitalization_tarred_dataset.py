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
import multiprocessing as mp
from pathlib import Path

from nemo.collections.nlp.data.token_classification.punctuation_capitalization_tarred_dataset import (
    create_tarred_dataset
)


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--text", "-t", type=Path, required=True)
    parser.add_argument("--labels", "-L", type=Path, required=True)
    parser.add_argument("--output_dir", "-o", type=Path, required=True)
    parser.add_argument("--max_seq_length", "-s", type=int, default=512)
    parser.add_argument("--tokens_in_batch", "-b", type=int, default=15000)
    parser.add_argument("--lines_per_dataset_fragment", type=int, default=10 ** 6)
    parser.add_argument("--num_batches_per_tarfile", type=int, default=1000)
    parser.add_argument("--tokenizer", "-T", default="bert-base-uncased")
    parser.add_argument("--tar_file_prefix", "-p", default="punctuation_capitalization")
    parser.add_argument("--n_jobs", "-j", type=int, default=mp.cpu_count())
    args = parser.parse_args()
    args.text = args.text.expanduser()
    args.labels = args.labels.expanduser()
    args.output_dir = args.output_dir.expanduser()
    return args


def main():
    args = get_args()
    create_tarred_dataset(
        args.text,
        args.labels,
        args.output_dir,
        args.max_seq_length,
        args.tokens_in_batch,
        args.lines_per_dataset_fragment,
        args.num_batches_per_tarfile,
        args.tokenizer,
        args.tar_file_prefix,
        args.n_jobs,
    )


if __name__ == "__main__":
    main()
