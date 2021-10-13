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
import os
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing as mp
from functools import partial

from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

#=============================================================================#
# Auxiliary methods
#=============================================================================#


def tokenize_line(line, tokenizer):
    """
    Returns a tokenized line
    """
    tokens = tokenizer.text_to_ids(line.decode("utf-8"))


def line_len(tokenizer, line):
    """
    Returns a tokenized length of a text line
    """
    tokens = tokenize_line(line, tokenizer)

    return len(tokens)


#=============================================================================#
# Main script
#=============================================================================#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collects statistics over tokenized dataset')
    parser.add_argument('input_files', metavar='N', type=str, nargs='+', help='Input files to parse')
    parser.add_argument('--tokenizer_library', type=str, required=True,
                        help='Path to pre-trained nemo-supported tokenizer model')
    parser.add_argument('--tokenizer_model', type=str, required=True,
                        help='Path to pre-trained nemo-supported tokenizer model')
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count(),
                        help='Number of workers (default to number of CPUs)')
    parser.add_argument('--max_lines', type=int, default=-1, help='Max number of lines to parse')
    parser.add_argument('--out_dir', type=str, default="", help='Path to store data and plots')

    args = parser.parse_args()

    tokenizer = get_nmt_tokenizer(
        library=args.tokenizer_library,
        tokenizer_model=args.tokenizer_model,
    )

    all_len = []

    for fn in args.input_files:
        print(f"Parsing fn = {fn}")
        # read file
        fh = open(fn)
        lines = [l.strip() for l in fh.readlines()]
        if args.max_lines > 0:
            lines = lines[:args.max_lines]

        # tokenize lines
        with mp.Pool(args.num_workers) as p:
            all_len.extend(p.map(partial(line_len, tokenizer=tokenizer), lines))

    # compute stats

    # save all results
    if args.out_dir:
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)

    stats = {
        "samples": len(all_len),
        "mean": np.mean(all_len),
        "stdev": np.std(all_len),
        "min": np.min(all_len),
        "max": np.max(all_len),
        "median": np.median(all_len),
    }

    print(f"stats = \n{stats}")

    # save all results
    if args.out_dir:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir, exist_ok=True)

        fh = open(os.path.join(args.out_dir, "lengths.txt"), "w")
        fw.writelines(["{l}\n".format(l=l) for l in all_len])

        fig = plt.hist(all_len)
        plt.savefig(os.path.join(args.out_dir, "lengths_hist.pdf"))
