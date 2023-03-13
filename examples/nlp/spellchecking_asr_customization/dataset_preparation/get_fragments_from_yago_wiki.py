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
This script contains logics for preparation of training examples from Yago Wikipedia preprocessed data.

Its input files are: 1) phrases, 2) normalized paragraphs. They should have equal number of lines.
It searches phrases in the paragraph, cut spans some phrase(s) and some surrounding context.

The goal is to get an incomplete "training example" which consists of:
   text span
   0, 1 or more correct candidates
   their exact positions in text

Its output is divided into two files: 1) examples with at least 1 correct candidate, 2) examples with no correct candidates.
This is done to allow for easy sampling in the future.
"""

import random
import re

from argparse import ArgumentParser
from typing import List, TextIO
from nemo.collections.nlp.data.spellchecking_asr_customization.utils import (
    CHARS_TO_IGNORE_REGEX,
    OOV_REGEX,
    preprocess_apostrophes_space_diacritics,
)

parser = ArgumentParser(description="Preparation of training examples from Yago Wikipedia preprocessed data")

# maskarada english the masquerade is the ninth studio album by serbian singer ceca it was released in nineteen ninety seven
parser.add_argument(
    "--input_paragraphs_file", required=True, type=str, help="Input file, each line is a text paragraph"
)

# masquerade;maskarada;ceca
# southampton;fortin;merchant's;medieval merchant's house
parser.add_argument(
    "--input_phrases_file",
    required=True,
    type=str,
    help="Input file, each line is one or multiple phrases separated by semicolon",
)

parser.add_argument(
    "--output_file_non_empty",
    required=True,
    type=str,
    help="Output file for fragments with at least 1 correct candidate",
)
parser.add_argument(
    "--output_file_empty", required=True, type=str, help="Output file for fragments with no correct candidates"
)
args = parser.parse_args()


def cut_spans(phrases: List[str], paragraph: str, out_non_empty: TextIO, out_empty: TextIO) -> None:
    p = preprocess_apostrophes_space_diacritics(paragraph)
    p_clean = CHARS_TO_IGNORE_REGEX.sub(" ", p).lower()  # number of characters is the same in p and p_clean
    p_clean = " ".join(p_clean.split(" "))

    p_clean_spaced = " " + p_clean + " "
    matches = []
    for phrase in phrases:
        pattern = " " + phrase + " "
        matches += list(re.finditer(pattern, p_clean_spaced))
    # sort found matches for all phrases by beginning
    sorted_matches = sorted(matches, key=lambda x: (x.start(), x.end()))
    space_matches = list(re.finditer(r"\s+", p_clean_spaced))  # these are already sorted by beginning

    next_phrase_match_id = 0 if len(sorted_matches) > 0 else -1
    for i in range(len(space_matches) - 1):
        begin = space_matches[i].start()
        j = random.randrange(min(i + 3, len(space_matches) - 1), min(i + 12, len(space_matches)))
        end = space_matches[j].start()
        # move next_phrase_match_id so that it cannot start before begin
        while next_phrase_match_id > -1 and sorted_matches[next_phrase_match_id].start() < begin:
            next_phrase_match_id += 1
            if next_phrase_match_id >= len(sorted_matches):
                next_phrase_match_id = -1
                break

        text = p_clean_spaced[begin + 1 : end]  # +1 to move from beginning space to next symbol
        if re.search(OOV_REGEX, text):
            print("bad text: ", text)
            continue

        targets = []
        phrase_match_id = next_phrase_match_id
        while phrase_match_id > -1:
            phrase_match = sorted_matches[phrase_match_id]
            if phrase_match.start() < end and phrase_match.end() <= end + 1:  # +1 because phrase ends with space
                targets.append((phrase_match.group(), phrase_match.start() - begin, phrase_match.end() - begin - 2))
                phrase_match_id += 1
                if phrase_match_id >= len(sorted_matches):
                    break
            else:
                break

        if len(targets) > 0:  # positive case
            target_str = " ".join(map(str, range(1, len(targets) + 1)))
            span_info = []
            for k, kstart, kend, in targets:
                span_info.append("[" + k.strip() + "] " + str(kstart) + " " + str(kend))
            out_non_empty.write(text + "\t" + target_str + "\t" + ";".join(span_info) + "\n")
        else:  # negative case
            out_empty.write(text + "\t0\t\n")


def main() -> None:
    random.seed(0)
    out_non_empty = open(args.output_file_non_empty, "w", encoding="utf-8")
    out_empty = open(args.output_file_empty, "w", encoding="utf-8")
    paragraphs_file = open(args.input_paragraphs_file, "r", encoding="utf-8")
    phrases_file = open(args.input_phrases_file, "r", encoding="utf-8")
    n = 0
    for line in paragraphs_file:
        paragraph = line.strip()
        phrases = phrases_file.readline().strip().split(";")
        cut_spans(phrases, paragraph, out_non_empty, out_empty)
        n += 1

    paragraphs_file.close()
    phrases_file.close()
    out_non_empty.close()
    out_empty.close()


if __name__ == "__main__":
    main()
