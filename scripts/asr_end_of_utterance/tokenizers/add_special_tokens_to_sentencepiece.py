# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import logging
import sys
import tempfile

from argparse import ArgumentParser
from pathlib import Path

import sentencepiece as spm

from nemo.core.connectors.save_restore_connector import SaveRestoreConnector

try:
    import sentencepiece_model_pb2 as spt
except (ImportError, ModuleNotFoundError):
    raise Exception("Ensure that sentencepiece_model_pb2.py has been generated from the protoc compiler")


SPECIAL_TOKENS = ["<EOU>", "<EOB>"]

"""Utility to add special tokens to existing sentencepiece models.

Generate sentencepiece_model_pb2.py in the directory of this script before running
To generate run `protoc --python_out=<path_to_NeMo>/scripts/asr_end_of_utterance/tokenizers sentencepiece_model.proto`
inside the src folder in sentencepiece repo
Refer: https://github.com/google/sentencepiece/issues/121

Usage:
python add_special_tokens_to_sentencepiece.py \
    --input_file your_model.nemo \
    --output_file /path/to/new/tokenizer.model
"""


parser = ArgumentParser(description="Add special tokens to sentencepiece model")
parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help="Path to sentencepiece model file",
)
parser.add_argument(
    "--output_file",
    type=str,
    required=True,
    help="Path to sentencepiece model file",
)
parser.add_argument(
    "--tokens",
    type=str,
    nargs='+',
    help="Special tokens to add to tokenizer",
    default=SPECIAL_TOKENS,
)
parser.add_argument(
    "--is_userdefined",
    action="store_true",
    help="When set, the new tokens are set as user_defined tokens",
)


def extract_nemo_tokenizer(nemo_filepath, output_dir):
    SaveRestoreConnector._unpack_nemo_file(path2file=nemo_filepath, out_folder=output_dir)
    tokenizer = None
    for file in Path(output_dir).glob("**/*"):
        if file.is_file() and file.name.endswith("tokenizer.model"):
            tokenizer = file
            break
    if tokenizer is None:
        raise ValueError(f"Tokenizer not found in {output_dir}: {os.listdir(output_dir)}")
    return str(tokenizer.absolute())


def edit_spt_model(input_file, output_file, tokens, is_userdefined):

    token_type = 3
    if is_userdefined:
        token_type = 4

    model = spt.ModelProto()
    model.ParseFromString(open(input_file, 'rb').read())

    for token in tokens:
        piece = model.SentencePiece(piece=token, score=0.0, type=token_type)
        if piece in model.pieces:
            logging.error(f"Special Token '{token}' already exists in the input model!")
            sys.exit(1)
        model.pieces.append(piece)

    sp = spm.SentencePieceProcessor()
    try:
        sp.LoadFromSerializedProto(model.SerializeToString())
        for token in tokens:
            id = sp.piece_to_id(token)
            logging.info(f"Created token '{token}' at ID {id}")
        logging.info(f"New tokenizer vocab size: {sp.get_piece_size()}")
    except:
        logging.error("Could not appropriately configure new tokenizer. Verify if the special tokens already exist.")
        sys.exit(1)

    with open(output_file, 'wb') as outf:
        outf.write(model.SerializeToString())

    logging.info(f"Created new tokenizer at: {output_file}")


def inject_special_tokens(input_file, output_file, tokens, is_userdefined):
    if not os.path.exists(input_file):
        raise ValueError(f"Input file {input_file} does not exist")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Check if input file is a Nemo file
        if input_file.endswith(".nemo"):
            input_file = extract_nemo_tokenizer(input_file, temp_dir)
            logging.info(f"Extracted tokenizer from Nemo file: {input_file}")
        else:
            input_file = os.path.abspath(input_file)
            logging.info(f"Using input file: {input_file}")

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(output_file):
            logging.info(f"Output file {output_file} already exists. Overwriting.")
        edit_spt_model(input_file, output_file, tokens, is_userdefined)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parser.parse_args()
    inject_special_tokens(args.input_file, args.output_file, args.tokens, args.is_userdefined)
