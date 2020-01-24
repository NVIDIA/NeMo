#!/usr/bin/env python

# =============================================================================
# Copyright 2019 AI Applications Design Team at NVIDIA. All Rights Reserved.
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
# =============================================================================

import argparse
import glob
import os

from sentencepiece import SentencePieceTrainer as SPT
from tqdm import tqdm

MERGED_FILE = "merged.txt"


def main():
    parser = argparse.ArgumentParser(description="Create vocabulary")
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--model_prefix", default="tokenizer", type=str)
    parser.add_argument("--num_placeholders", default=100, type=int)
    parser.add_argument("--sample_size", default=1e7, type=int)
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--vocab_filename", default="vocab.txt", type=str)
    parser.add_argument("--vocab_size", default=32000, type=int)
    args = parser.parse_args()

    if args.dataset_dir is not None and args.train_path is not None:
        logging.info("Only one of 'dataset_dir' and 'train_path' can be specified")
        return
    elif args.dataset_dir is not None:
        # If the dataset is distributed across multiple files, merge into one
        # file before proceeding
        # filepaths = glob.glob(os.path.join(args.dataset_dir, "**", "*.txt"))
        filepaths = glob.glob(os.path.join(args.dataset_dir, "*.txt"))
        logging.info("Found {} files, concatenenating dataset into one file..."
              .format(len(filepaths)))

        with open(MERGED_FILE, "w") as f:
            for filepath in tqdm(filepaths):
                f.write(open(filepath, "r", errors="ignore").read())

        train_path = MERGED_FILE
    elif args.train_path is not None:
        train_path = args.train_path
    else:
        logging.info("One of 'dataset_dir' and 'train_path' must be specified")
        return

    SPT.Train("--input={} ".format(train_path) +
              "--model_prefix={} ".format(args.model_prefix) +
              "--vocab_size={} ".format(args.vocab_size
                                        - args.num_placeholders) +
              "--input_sentence_size={} ".format(args.sample_size) +
              "--shuffle_input_sentence=true " +
              "--hard_vocab_limit=false " +
              "--bos_id=-1 " +
              "--eos_id=-1")

    # Add BERT control symbols
    vocab = ["[PAD]"]
    tokens = []

    with open("{}.vocab".format(args.model_prefix), "r") as f:
        # Skip first <unk> token
        f.seek(8)

        # Read tokens from each line and parse for vocab
        for line in f:
            piece = line.split("\t")[0]

            if piece.startswith("▁"):
                token = piece[1:]
            else:
                token = "##{}".format(piece)

            tokens.append(token)

    vocab.extend(["[unused{}]".format(i)
                 for i in range(args.vocab_size - len(tokens))])
    vocab.extend(["[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    vocab.extend(tokens)

    # Save vocabulary to output file
    with open(args.vocab_filename, "w") as f:
        for token in vocab:
            f.write("{}\n".format(token))


if __name__ == "__main__":
    main()
