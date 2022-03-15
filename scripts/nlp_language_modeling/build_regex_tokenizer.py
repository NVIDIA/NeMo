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
"""
python -- build_regex_tokenizer.py \
  --regex '\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]' \
  --input_file text_data_file.txt \
  --input_type text \
  --output_file regex_tokenizer
"""
import argparse

from nemo.utils import logging
from nemo.collections.common.tokenizers.regex_tokenizer import RegExTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Builds vocabulary from regex tokenizer. Outputs .model (regular expression) and .vocab (learned vocabualry)",
    )
    parser.add_argument(
        '--regex', type=str, required=True, help='Regular expression to split text',
    )
    parser.add_argument(
        '--input_file', type=str, required=True, help='Input text/csv file',
    )
    parser.add_argument(
        '--output_file',
        type=str,
        required=True,
        help='Output base file name. Two files will be created: .vocab (learned vocabulary), .model (the regex)',
    )
    parser.add_argument(
        '--input_type',
        type=str,
        required=False,
        choices=["text", "csv"],
        default="text",
        help='Type of input file: text, csv',
    )
    parser.add_argument(
        '--input_csv_col',
        type=str,
        required=False,
        default="smiles",
        help='Column of data in CSV file',
    )
    args = parser.parse_args()

    model_fname = args.output_file + ".model"
    vocab_fname = args.output_file + ".vocab"

    # save .model with regex string
    logging.info(f"Saving regex in model file: {model_fname}")
    with open(model_fname, 'w') as fp:
        fp.write(args.regex)

    # learn vocabulary and save to .vocab
    logging.info(f"Saving vocabulary in file: {vocab_fname}")
    if args.input_type == "csv":
        RegExTokenizer.create_vocab_from_csv(
            data_csv_file=model_fname,
            vocab_file=vocab_fname,
            regex=args.regex,
            col=args.input_csv_col,
        )
    elif args.input_type == "text":
        RegExTokenizer.create_vocab_from_text(
            data_text_file=model_fname,
            vocab_file=vocab_fname,
            regex=args.regex,
        )
