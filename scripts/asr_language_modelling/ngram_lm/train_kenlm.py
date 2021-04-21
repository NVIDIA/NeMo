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
import logging
import os
import subprocess
import sys

import torch
from kenlm_text_utils import read_train_file, tokenize_text

import nemo.collections.asr as nemo_asr

"""
NeMo's beam search decoders only support char-level encodings. In order to make it work with BPE-level encodings, we
use a trick to encode the sub-word tokens of the training data as unicode characters and train a char-level KenLM. 
TOKEN_OFFSET is the offset in the unicode table to be used to encode the BPE sub-words.

This encoding scheme reduces the required memory significantly, and the LM and its binary blob format require 
less storage space. The only drawback is that there is a symmetric decoding by this offset value required when 
performing actual beam search, but it is negligible in compute cost and the storage space benefits are more useful.
"""
TOKEN_OFFSET = 100

CHUNK_SIZE = 8192
CHUNK_BUFFER_SIZE = 512


def main():
    parser = argparse.ArgumentParser(
        description='Train an N-gram language model with KenLM to be used with beam search decoder of ASR models.'
    )
    parser.add_argument(
        "--train_file",
        required=True,
        type=str,
        help="Path to the training file. It can be a text file or Json manifest",
    )
    parser.add_argument(
        "--nemo_model_file", required=True, type=str, help="The path of the '.nemo' file of the ASR model"
    )
    parser.add_argument(
        "--kenlm_model_file", required=True, type=str, help="The path to store the KenLM binary model file"
    )
    parser.add_argument("--ngram_length", required=True, type=int, help="The order of N-gram LM")
    parser.add_argument("--kenlm_bin_path", required=True, type=str, help="The path to the bin folder of KenLM")
    parser.add_argument(
        "--do_lowercase", action='store_true', help="Whether to apply lower case conversion on the trainig text"
    )
    args = parser.parse_args()

    """ TOKENIZER SETUP """
    logging.info(f"Loading nemo model '{args.nemo_model_file}' ...")
    model = nemo_asr.models.EncDecCTCModelBPE.restore_from(args.nemo_model_file, map_location=torch.device('cpu'))

    encoded_train_file = f"{args.kenlm_model_file}.tmp.txt"
    """ DATASET SETUP """
    logging.info(f"Encoding the train file '{args.train_file}' ...")
    dataset = read_train_file(args.train_file, lowercase=args.do_lowercase)
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
    # --discount_fallback is needed for training KenLM for BPE-based models
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
    subprocess.run(kenlm_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr)

    """ BINARY BUILD """
    logging.info(f"Running binary_build command \n\n{' '.join(kenlm_args)}\n\n")
    kenlm_args = [
        os.path.join(args.kenlm_bin_path, "build_binary"),
        "trie",
        f"{args.kenlm_model_file}.tmp.arpa",
        args.kenlm_model_file,
    ]
    subprocess.run(kenlm_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr)

    os.remove(encoded_train_file)


if __name__ == '__main__':
    main()
