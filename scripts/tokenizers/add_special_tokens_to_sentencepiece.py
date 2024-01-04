# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import sys
from argparse import ArgumentParser

import sentencepiece as spm

try:
    import sentencepiece_model_pb2 as spt
except (ImportError, ModuleNotFoundError):
    raise Exception("Ensure that sentencepiece_model_pb2.py has been generated from the protoc compiler")


"""Utility to add special tokens to existing sentencepiece models.

Generate sentencepiece_model_pb2.py in the directory of this script before running
To generate run `protoc --python_out=<path_to_NeMo>/scripts/tokenizers/ sentencepiece_model.proto`
inside the src folder in sentencepiece repo
Refer: https://github.com/google/sentencepiece/issues/121

Usage:
python edit_spt_model.py \
    --input_file <input_model_dir> \
    --output_file <output_model_dir> \
    --tokens <space separated special tokens>

Example:
python edit_spt_model.py \
    --input_file test.model \
    --output_file test.model \
    --tokens [CLS] [SEP]
"""


def edit_spt_model():
    parser = ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to sentencepiece model file",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to sentencepiece model file",
    )
    parser.add_argument(
        "--tokens", type=str, nargs='+', required=True, help="Special tokens to add to tokenizer",
    )
    parser.add_argument(
        "--is_userdefined", action="store_true", help="When set, the new tokens are set as user_defined tokens",
    )
    args = parser.parse_args()

    token_type = 3
    if args.is_userdefined:
        token_type = 4

    model = spt.ModelProto()
    model.ParseFromString(open(args.input_file, 'rb').read())

    tokens = ['<cls>', '<sep>', '<mask>' ,'<bos>', 'p{ }', 'p{!}', 'p{"}', 'p{#1}', 'p{#2}', 'p{#3}', 'p{#4}', 'p{#5}', "p{'}", 'p{(}', 'p{)}', 'p{,}', 'p{-}', 'p{.}', 'p{/}', 'p{:}', 'p{;}', 'p{?}', 'p{A}', 'p{B}', 'p{C}', 'p{D}', 'p{E}', 'p{F}', 'p{G}', 'p{H}', 'p{I}', 'p{J}', 'p{K}', 'p{L}', 'p{M}', 'p{N}', 'p{O}', 'p{P}', 'p{Q}', 'p{R}', 'p{S}', 'p{T}', 'p{U}', 'p{V}', 'p{W}', 'p{X}', 'p{Y}', 'p{Z}', 'p{[}', 'p{]}', 'p{ai}', 'p{au}', 'p{a}', 'p{b}', 'p{d}', 'p{ei}', 'p{e}', 'p{f}', 'p{h}', 'p{i}', 'p{j}', 'p{k}', 'p{kʰ}', 'p{l}', 'p{m}', 'p{n}', 'p{ou}', 'p{o}', 'p{p}', 'p{pʰ}', 'p{r}', 'p{s}', 'p{ts}', 'p{tsʰ}', 'p{t}', 'p{tɕ}', 'p{tɕʰ}', 'p{tʰ}', 'p{u}', 'p{v}', 'p{w}', 'p{x}', 'p{y}', 'p{z}', 'p{{}', 'p{}}', 'p{\xa0}', 'p{¡}', 'p{«}', 'p{»}', 'p{¿}', 'p{Á}', 'p{É}', 'p{Í}', 'p{Ñ}', 'p{Ó}', 'p{Ú}', 'p{Ü}', 'p{æ}', 'p{ð}', 'p{ø}', 'p{ŋ}', 'p{œ}', 'p{ɐ}', 'p{ɑ}', 'p{ɒ}', 'p{ɔ}', 'p{ɕ}', 'p{ə}', 'p{ɚ}', 'p{ɛ}', 'p{ɜ}', 'p{ɝ}', 'p{ɡ}', 'p{ɣ}', 'p{ɤ}', 'p{ɥ}', 'p{ɪ}', 'p{ɬ}', 'p{ɲ}', 'p{ɹ}', 'p{ɾ}', 'p{ʁ}', 'p{ʂ}', 'p{ʃ}', 'p{ʈʂ}', 'p{ʈʂʰ}', 'p{ʊ}', 'p{ʌ}', 'p{ʎ}', 'p{ʐ}', 'p{ʒ}', 'p{ʔ}', 'p{ʝ}', 'p{ʲ}', 'p{ˈ}', 'p{ˌ}', 'p{ː}', 'p{̀}', 'p{́}', 'p{̂}', 'p{̃}', 'p{̈}', 'p{̧}', 'p{̩}', 'p{β}', 'p{θ}', 'p{ᵻ}', 'p{–}', 'p{“}', 'p{”}', 'p{…}', 'p{‹}', 'p{›}', 'p{、}', 'p{。}', 'p{《}', 'p{》}', 'p{！}', 'p{（}', 'p{）}', 'p{，}', 'p{：}', 'p{；}', 'p{？}']

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

    with open(args.output_file, 'wb') as outf:
        outf.write(model.SerializeToString())

    logging.info(f"Created new tokenizer at: {args.output_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    edit_spt_model()
