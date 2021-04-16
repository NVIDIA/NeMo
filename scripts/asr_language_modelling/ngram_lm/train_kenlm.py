# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
#

import argparse
import gc
import json
import logging
import os
import subprocess
import sys

import torch
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import nemo.collections.asr as nemo_asr

CHUNK_SIZE = 8192
CHUNK_BUFFER_SIZE = 512
TOKEN_OFFSET = 100


def read_text(path, lowercase: bool = False):
    lines_read = 0
    text_dataset = []

    # Roughly 40 Million lines of text
    with open(path, 'r') as f:
        reader = tqdm(iter(lambda: f.readline(), ''), desc="Read 0 lines", unit=' lines')
        for i, line in enumerate(reader):
            if path.endswith('.json'):
                line = json.loads(line)['text']

            line = line.replace("\n", "").strip()
            if lowercase:
                line = line.lower()

            if line:
                text_dataset.append(line)

                lines_read += 1
                if lines_read % 100000 == 0:
                    reader.set_description(f"Read {lines_read} lines")

    return text_dataset


def tokenize_str(texts, tokenizer, ids_to_tokens, offset):
    tokenized_text = []
    for text in texts:
        tok_text = tokenizer.text_to_ids(text)
        tok_text = [chr(token + offset) for token in tok_text]
        # tok_text = [ids_to_tokens[idx] for idx in tok_text]
        tokenized_text.append(tok_text)
    return tokenized_text


def tokenize_text(data, tokenizer, path, chunk_size=8192, buffer_size=32, token_offset=100):
    dataset_len = len(data)
    print(
        f"Chunking {dataset_len} rows into {dataset_len / float(chunk_size):0.4f} tasks (each chunk contains {chunk_size} elements)"
    )

    vocabulary = tokenizer.tokenizer.get_vocab()
    idx_to_token = {v: k for k, v in vocabulary.items()}

    current_step = 0

    if os.path.exists(path):
        print(f"Deleting previous file : {path}")
        os.remove(path)

    with Parallel(n_jobs=-2, verbose=10) as parallel:
        while True:
            start = current_step * chunk_size
            end = min((current_step + buffer_size) * chunk_size, dataset_len)

            tokenized_data = parallel(
                delayed(tokenize_str)(data[start : start + chunk_size], tokenizer, idx_to_token, token_offset)
                for start in range(start, end, chunk_size)
            )

            # Write dataset
            write_dataset(tokenized_data, path)

            current_step += len(tokenized_data)

            print(f"Finished writing {len(tokenized_data)} chunks to {path}. Current chunk index = {current_step}")

            del tokenized_data
            gc.collect()

            if end >= dataset_len:
                break


def write_dataset(chunks, path):
    basedir = os.path.dirname(path)

    if not os.path.exists(basedir):
        os.makedirs(basedir, exist_ok=True)

    with open(path, 'a+', encoding='utf-8') as f:
        for chunk_idx in tqdm(range(len(chunks)), desc='Chunk ', total=len(chunks), unit=' chunks'):
            for text in chunks[chunk_idx]:
                line = ' '.join(text)
                f.write(f"{line}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train an N-gram language model with KenLM to be used with beam search decoder of ASR models.'
    )
    parser.add_argument(
        "--train_file",
        required=True,
        type=str,
        help="Path to the training file. It can be a text file or Json manifest.",
    )
    parser.add_argument("--nemo_model_file", required=True, type=str)
    parser.add_argument("--kenlm_model_file", required=True, type=str)
    parser.add_argument("--ngram_length", required=True, type=int)
    parser.add_argument("--do_lowercase", action='store_true')
    parser.add_argument("--kenlm_bin_path", required=True, type=str)
    args = parser.parse_args()

    """ TOKENIZER SETUP """
    logging.info(f"Loading nemo model '{args.nemo_model_file}' ...")
    model = nemo_asr.models.EncDecCTCModelBPE.restore_from(args.nemo_model_file, map_location=torch.device('cpu'))

    encoded_train_file = f"{args.kenlm_model_file}.tmp.txt"
    """ DATASET SETUP """
    logging.info(f"Encoding the train file '{args.train_file}' ...")
    dataset = read_text(args.train_file, lowercase=args.do_lowercase)
    tokenize_text(
        dataset,
        model.tokenizer,
        path=encoded_train_file,
        chunk_size=CHUNK_SIZE,
        buffer_size=CHUNK_BUFFER_SIZE,
        token_offset=TOKEN_OFFSET,
    )

    del model

    """ LMPLZ ARGUMENT SETUP """
    kenlm_args = [
        os.path.join(args.kenlm_bin_path, 'lmplz'),
        "-o",
        f"{args.ngram_length}",
        "--text",
        encoded_train_file,
        "--arpa",
        f"{args.kenlm_model_file}.tmp.arpa",
        "--discount_fallback",
    ]
    ret = subprocess.run(kenlm_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr)

    """ BINARY BUILD """
    logging.info(f"Running binary_build command \n\n{' '.join(kenlm_args)}\n\n")
    kenlm_args = [
        os.path.join(args.kenlm_bin_path, "build_binary"),
        "trie",
        f"{args.kenlm_model_file}.tmp.arpa",
        args.kenlm_model_file,
    ]
    ret = subprocess.run(kenlm_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr)

    os.remove(encoded_train_file)


if __name__ == '__main__':
    main()
