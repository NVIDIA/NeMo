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
    parser = argparse.ArgumentParser(
        description="Create vocabulary for a BERT-like model")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_prefix", default="tokenizer", type=str)
    parser.add_argument("--num_placeholders", default=0, type=int)
    parser.add_argument("--sample_size", default=1e7, type=int)
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--vocab_filename", default="vocab.txt", type=str)
    parser.add_argument("--vocab_size", default=32000, type=int)
    args = parser.parse_args()

    data_path = args.data_path
    if not os.path.exists(data_path):
        raise ValueError(f"{data_path} doesn't exist.")

    if os.path.isdir(data_path):
        files = glob.glob(f'{data_path}/*.txt')
        print(f"Concatenenating {len(filepaths)} txt files into {MERGED_FILE}")

        with open(MERGED_FILE, "w") as merged:
            for file in tqdm(files):
                with open(file, 'r') as inf:
                    content = inf.read().strip()
                merged.write(content + '\n\n\n')
        data_path = f'{data_path}/{MERGED_FILE}'

    cmd = (f"--input={data_path} --model_prefix={args.model_prefix} "
           f"--vocab_size={args.vocab_size - args.num_placeholders} "
           f"--input_sentence_size={args.sample_size} "
           f"--shuffle_input_sentence=true --hard_vocab_limit=false "
           f"--bos_id=-1 --eos_id=-1")
    SPT.Train(cmd)

    # Add BERT control symbols
    vocab = ["[PAD]"]
    tokens = []

    with open(f"{args.model_prefix}.vocab", "r") as f:
        f.readline()  # skip first <unk> token

        # Read tokens from each line and parse for vocab
        for line in f:
            piece = line.split("\t")[0]
            token = piece[1:] if piece.startswith("▁") else f"##{piece}"
            tokens.append(token)

    vocab.extend([f"[unused{i}]" for i in range(args.vocab_size - len(tokens))])
    vocab.extend(["[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    vocab.extend(tokens)

    # Save vocabulary to output file
    with open(args.vocab_filename, "w") as f:
        for token in vocab:
            f.write(f"{token}\n".format())


if __name__ == "__main__":
    main()
