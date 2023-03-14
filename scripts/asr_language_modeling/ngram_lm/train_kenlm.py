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


# This script would train an N-gram language model with KenLM library (https://github.com/kpu/kenlm) which can be used
# with the beam search decoders on top of the ASR models. This script supports both character level and BPE level
# encodings and models which is detected automatically from the type of the model.
# After the N-gram model is trained, and stored in the binary format, you may use
# 'scripts/ngram_lm/eval_beamsearch_ngram.py' to evaluate it on an ASR model.
#
# You need to install the KenLM library and also the beam search decoders to use this feature. Please refer
# to 'scripts/ngram_lm/install_beamsearch_decoders.sh' on how to install them.
#
# USAGE: python train_kenlm.py --tokenizer_model_file <path to the .nemo file of the model> \
#                              --train_path <path to the training text or JSON manifest file \
#                              --kenlm_bin_path <path to the bin folder of KenLM library> \
#                              --kenlm_model_file <path to store the binary KenLM model> \
#                              --ngram_length <order of N-gram model>
#
# After training is done, the binary LM model is stored at the path specified by '--kenlm_model_file'.
# You may find more info on how to use this script at:
# https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_language_modeling.html

import argparse
import logging
import os
import subprocess
import sys
from glob import glob

import kenlm_utils
from nemo.utils import logging

"""
NeMo's beam search decoders only support char-level encodings. In order to make it work with BPE-level encodings, we
use a trick to encode the sub-word tokens of the training data as unicode characters and train a char-level KenLM. 
TOKEN_OFFSET is the offset in the unicode table to be used to encode the BPE sub-words. This encoding scheme reduces 
the required memory significantly, and the LM and its binary blob format require less storage space.
"""
TOKEN_OFFSET = kenlm_utils.TOKEN_OFFSET
CHUNK_SIZE = 8192
CHUNK_BUFFER_SIZE = 512


def main():
    file_path = os.path.split(os.path.realpath(__file__))[0]
    parser = argparse.ArgumentParser(
        description='Train an N-gram language model with KenLM to be used with beam search decoder of ASR models.'
    )
    parser.add_argument(
        "--train_path",
        required=True,
        nargs="+",
        type=str,
        help="Path to the training files or whitespace separated files. Files can be a text file, JSON manifest or .json.gz",
    )

    parser.add_argument(
        "--tokenizer_model_file",
        required=True,
        type=str,
        help="The path to '.model' file of the SentencePiece tokenizer, or '.nemo' file of the ASR model, or name of a pretrained NeMo model",
    )
    parser.add_argument(
        "--kenlm_model_file", required=True, type=str, help="The path to store the KenLM binary model file"
    )
    parser.add_argument("--ngram_length", required=True, type=int, help="The order of N-gram LM")
    parser.add_argument(
        "--ngram_prun",
        default="0",
        required=False,
        nargs="+",
        type=str,
        help="Whitespace separated digits. Example: 0 0 1",
    )
    parser.add_argument("--kenlm_bin_path", required=True, type=str, help="The path to the bin folder of KenLM")
    parser.add_argument("--cache_path", required=False, type=str, default=None, help="Cache path")
    parser.add_argument(
        "--do_lowercase", action='store_true', help="Whether to apply lower case conversion on the training text"
    )
    parser.add_argument("--clean_text", action='store_true', help="Whether to clean the text")
    parser.add_argument(
        "--punctuation_to_preserve",
        required=False,
        default='',
        type=str,
        help="Punctuation marks to preserve in text when --clean_text is used",
    )
    parser.add_argument(
        "--separate_punctuation",
        action='store_true',
        help="Whether to separate punctuation with the previouse word by space when --clean_text and --punctuation_to_preserveis is used",
    )
    parser.add_argument("--verbose", type=int, default=0, help="Verbose level from 0. Default is 1 ")

    args = parser.parse_args()

    assert (
        args.clean_text or (not args.punctuation_to_preserve) and (not args.clean_text)
    ), "--punctuation_to_preserve work only with --clean_text "
    assert (
        args.clean_text or (not args.punctuation_to_preserve) and (not args.clean_text) and (not args.separate_punctuation)
    ), "--separate_punctuation work only with --clean_text and --punctuation_to_preserve"
    args.train_path = kenlm_utils.get_train_list(args.train_path)

    if type(args.ngram_prun) == str:
        args.ngram_prun = [args.ngram_prun]

    tokenizer, encoding_level, offset_encoding = kenlm_utils.setup_tokenizer(args.tokenizer_model_file)
    encoded_train_files = []

    if args.cache_path:
        """ DATASET SETUP """

        for file_num, train_file in enumerate(args.train_path):
            logging.info(
                f"Encoding the train file '{train_file}' number {file_num+1} out of {len(args.train_path)} ..."
            )

            cached_files = glob(os.path.join(args.cache_path, os.path.split(train_file)[1]) + "*")
            encoded_train_file = os.path.join(args.cache_path, os.path.split(train_file)[1] + f"_{file_num}.tmp.txt")
            if cached_files:
                if cached_files[0] != encoded_train_file:
                    os.rename(cached_files[0], encoded_train_file)
                    logging.info("Rename", cached_files[0], "to", encoded_train_file)
            else:
                dataset = kenlm_utils.read_train_file(
                    train_file,
                    do_lowercase=args.do_lowercase,
                    punctuation_to_preserve=args.punctuation_to_preserve,
                    separate_punctuation=args.separate_punctuation,
                    clean_text=args.clean_text,
                    verbose=args.verbose,
                )

                if encoding_level == "subword":
                    kenlm_utils.tokenize_text(
                        dataset,
                        tokenizer,
                        path=encoded_train_file,
                        chunk_size=CHUNK_SIZE,
                        buffer_size=CHUNK_BUFFER_SIZE,
                        token_offset=TOKEN_OFFSET if offset_encoding else -1,
                    )
                else:
                    with open(encoded_train_file, 'w', encoding='utf-8') as f:
                        for line in dataset:
                            f.write(f"{line}\n")

            encoded_train_files.append(encoded_train_file)

        first_process_args = ["cat"] + encoded_train_files

    else:
        first_process_args = [
            "python3",
            os.path.join(file_path, "kenlm_utils.py"),
            "--tokenizer_model_file",
            args.tokenizer_model_file,
            "--punctuation_to_preserve",
            args.punctuation_to_preserve,
            "--train_path",
        ] + args.train_path
        if args.clean_text:
            first_process_args.append("--clean_text")
        if args.do_lowercase:
            first_process_args.append("--do_lowercase")
        if args.separate_punctuation:
            first_process_args.append("--separate_punctuation")

    first_process = subprocess.Popen(first_process_args, stdout=subprocess.PIPE)

    if encoding_level == "subword":
        discount_arg = "--discount_fallback"  # --discount_fallback is needed for training KenLM for BPE-based models
    else:
        discount_arg = ""
    arpa_file = f"{args.kenlm_model_file}.tmp.arpa"
    """ LMPLZ ARGUMENT SETUP """
    kenlm_args = [
        os.path.join(args.kenlm_bin_path, 'lmplz'),
        "-o",
        str(args.ngram_length),
        "--arpa",
        arpa_file,
        discount_arg,
        "--prune",
    ] + args.ngram_prun

    ret = subprocess.run(
        kenlm_args, stdin=first_process.stdout, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr
    )
    first_process.wait()
    logging.setLevel(logging.INFO)
    if ret.returncode != 0:
        raise RuntimeError("Training KenLM was not successful!")
    """ BINARY BUILD """
    logging.info(f"Running binary_build command \n\n{' '.join(kenlm_args)}\n\n")
    kenlm_args = [
        os.path.join(args.kenlm_bin_path, "build_binary"),
        "trie",
        arpa_file,
        args.kenlm_model_file,
    ]
    ret = subprocess.run(kenlm_args, capture_output=False, text=True, stdout=sys.stdout, stderr=sys.stderr)

    if ret.returncode != 0:
        raise RuntimeError("Training KenLM was not successful!")


if __name__ == '__main__':
    main()
