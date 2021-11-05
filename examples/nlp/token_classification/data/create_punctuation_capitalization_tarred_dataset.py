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
    build_label_ids_from_list_of_labels, create_tarred_dataset
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
    parser.add_argument("--tokenizer_name", "-T", default="bert-base-uncased")
    parser.add_argument("--tokenizer_model", "-m", type=Path)
    parser.add_argument("--vocab_file", "-v", type=Path)
    parser.add_argument("--merges_file", "-M", type=Path)
    parser.add_argument("--special_token_names", "-n", nargs="+")
    parser.add_argument("--special_token_values", "-V", nargs="+")
    parser.add_argument("--use_fast_tokenizer", "-f", action="store_true")
    parser.add_argument("--tokenizer_bpe_dropout", "-d", type=float)
    parser.add_argument("--pad_label", "-P", default='O', help="Pad label both for punctuation and capitalization.")
    parser.add_argument("--punct_labels", "-p", nargs="+", help="All punctuation labels EXCEPT PAD LABEL.")
    parser.add_argument("--capit_labels", "-c", nargs="+", help="All capitalization labels EXCEPT PAD LABEL.")
    parser.add_argument("--punct_label_ids_file", type=Path)
    parser.add_argument("--capit_label_ids_file", type=Path)
    parser.add_argument("--tar_file_prefix", "-x", default="punctuation_capitalization")
    parser.add_argument("--n_jobs", "-j", type=int, default=mp.cpu_count())
    args = parser.parse_args()
    for name in [
        "text",
        "labels",
        "output_dir",
        "tokenizer_model",
        "vocab_file",
        "merges_file",
        "punct_label_ids_file",
        "capit_label_ids_file"
    ]:
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser())
    if args.special_token_names is not None or args.special_token_values is not None:
        if args.special_token_names is None:
            raise parser.error(
                "If you provide parameter `--special_token_values` you have to provide parameter "
                "`--special_token_names`."
            )
        if args.special_token_values is None:
            raise parser.error(
                "If you provide parameter `--special_token_names` you have to provide parameter "
                "`--special_token_values`."
            )
        if len(args.special_token_names) != len(args.special_token_values):
            raise parser.error(
                f"Parameters `--special_token_names` and `--special_token_values` have to have equal number of values "
                f"whereas parameter `--special_token_names` has {len(args.special_token_names)} values and parameter "
                f"`--special_token_values` has {len(args.special_token_values)} values."
            )
        if len(set(args.special_token_names)) != len(args.special_token_names):
            for i in range(len(args.special_token_names) - 1):
                if args.special_token_names[i] in args.special_token_names[i + 1 :]:
                    raise parser.error(
                        f"Values of parameter `--special_token_names` has to be unique. Found duplicate value "
                        f"'{args.special_token_names[i]}'."
                    )
    return args


def main():
    args = get_args()
    if args.special_token_names is None:
        special_tokens = None
    else:
        special_tokens = dict(zip(args.special_token_names, args.special_token_values))

    if args.punct_labels is not None:
        punct_label_ids = build_label_ids_from_list_of_labels(args.pad_label, args.punct_labels)
    else:
        punct_label_ids = None

    if args.capit_labels is not None:
        capit_label_ids = build_label_ids_from_list_of_labels(args.pad_label, args.capit_labels)
    else:
        capit_label_ids = None

    create_tarred_dataset(
        args.text,
        args.labels,
        args.output_dir,
        args.max_seq_length,
        args.tokens_in_batch,
        args.lines_per_dataset_fragment,
        args.num_batches_per_tarfile,
        args.tokenizer_name,
        tokenizer_model=args.tokenizer_model,
        vocab_file=args.vocab_file,
        merges_file=args.merges_file,
        special_tokens=special_tokens,
        use_fast_tokenizer=args.use_fast_tokenizer,
        tokenizer_bpe_dropout=args.tokenizer_bpe_dropout,
        pad_label=args.pad_label,
        punct_label_ids=punct_label_ids,
        capit_label_ids=capit_label_ids,
        punct_label_ids_file=args.punct_label_ids_file,
        capit_label_ids_file=args.capit_label_ids_file,
        tar_file_prefix=args.tar_file_prefix,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
