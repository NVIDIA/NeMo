# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This script is used to get frequent ngrams from Yago Wikipedia preprocessed data.
Input file is normalized paragraphs.

The goal is to get a list of frequent ngrams up to given maximum length.
"""

import re
from argparse import ArgumentParser
from typing import Dict, List

from nemo.collections.nlp.data.spellchecking_asr_customization.utils import (
    CHARS_TO_IGNORE_REGEX,
    OOV_REGEX,
    preprocess_apostrophes_space_diacritics,
)

parser = ArgumentParser(description="Collecting frequent ngrams from Yago Wikipedia preprocessed data")

# maskarada english the masquerade is the ninth studio album by serbian singer ceca it was released in nineteen ninety seven
parser.add_argument(
    "--input_file", required=True, type=str, help="Input file, each line is a normalized text paragraph"
)

parser.add_argument("--output_file", required=True, type=str, help="Output file prefix")
parser.add_argument("--max_ngram_len", required=True, type=int, help="Maximum ngram length")
parser.add_argument("--min_freq", required=True, type=int, help="Minimum ngram frequency")
args = parser.parse_args()


def collect_ngrams(ngrams: List[Dict[str, int]], length: int, min_freq: int) -> None:
    print("Collecting ngrams of length ", length)
    paragraphs_file = open(args.input_file, "r", encoding="utf-8")
    tmp_vocab = {}
    for line in paragraphs_file:
        paragraph = line.strip()
        p = preprocess_apostrophes_space_diacritics(paragraph)
        p_clean = CHARS_TO_IGNORE_REGEX.sub(" ", p).lower()  # number of characters is the same in p and p_clean
        p_clean = " " + " ".join(p_clean.split(" ")) + " "

        space_matches = list(re.finditer(r"\s+", p_clean))  # these are already sorted by beginning

        if length == 1:
            for i in range(len(space_matches) - length):
                begin = space_matches[i].start()
                end = space_matches[i + length].start()
                ngram = p_clean[begin + 1 : end]
                if re.search(OOV_REGEX, ngram):
                    continue
                if ngram not in tmp_vocab:
                    tmp_vocab[ngram] = 0
                tmp_vocab[ngram] += 1
        else:
            for i in range(len(space_matches) - length):
                begin = space_matches[i].start()
                prefix_end = space_matches[i + length - 1].start()
                prefix_ngram = p_clean[begin + 1 : prefix_end]  # +1 to move from beginning space to next symbol
                if prefix_ngram not in ngrams[length - 1]:
                    continue
                end = space_matches[i + length].start()
                ngram = p_clean[begin + 1 : end]
                if re.search(OOV_REGEX, ngram):
                    continue
                if ngram not in tmp_vocab:
                    tmp_vocab[ngram] = 0
                tmp_vocab[ngram] += 1

    # delete too rare ngrams
    for ngram in tmp_vocab:
        if tmp_vocab[ngram] >= min_freq:
            ngrams[length][ngram] = tmp_vocab[ngram]

    paragraphs_file.close()


def main() -> None:
    ngrams = [{} for i in range(args.max_ngram_len + 1)]  # dict with index 0 won't be used

    for i in range(1, len(ngrams)):
        collect_ngrams(ngrams, length=i, min_freq=args.min_freq)

    for i in range(1, len(ngrams)):
        with open(args.output_file + "." + str(i), "w", encoding="utf-8") as out:
            for k in ngrams[i]:
                out.write(k + "\t" + str(ngrams[i][k]) + "\n")


if __name__ == "__main__":
    main()
