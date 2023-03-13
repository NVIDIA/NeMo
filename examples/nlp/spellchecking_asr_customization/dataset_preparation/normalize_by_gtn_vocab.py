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
This script can be used to perform a simple text normalization using phrase vocabulary from Google Text Normalization dataset.
"""

import re
from argparse import ArgumentParser
from typing import Dict, Tuple

from nemo.collections.nlp.data.spellchecking_asr_customization.utils import preprocess_apostrophes_space_diacritics

parser = ArgumentParser(description="Text Normalization Data Preprocessing for English")
parser.add_argument("--input_file", required=True, type=str, help="Input file, each line is a text paragraph")
parser.add_argument("--tn_vocab", required=True, type=str, help="Multi variant vocabulary")
parser.add_argument("--output_file", required=True, type=str, help="Output file")
args = parser.parse_args()

CHARS_TO_IGNORE = "\.\,\?\:\-!;()«»…\]\[/\*–‽+&_\\>€™$•}{~—=“\"”″‟„'"


def process_paragraph(paragraph: str, tn_vocab: Dict[str, Tuple[str, int]]) -> str:
    normalized_paragraph = ""
    pseudo_words = paragraph.split(" ")  # a pseudo word can have some punctuation attached
    mask = [0] * len(pseudo_words)
    for begin in range(len(pseudo_words)):
        if mask[begin]:  # if this position had been already replaced as part of some phrase
            continue
        for end in range(min(begin + 6, len(pseudo_words) + 1), begin, -1):  # loop from longest to shortest
            phrase = " ".join(pseudo_words[begin:end])
            if phrase in tn_vocab:
                spoken, _ = tn_vocab[phrase]
                normalized_paragraph += spoken + " "
                mask[begin:end] = [1] * (end - begin)
                break  # exit loop after success, in favor of longer phrase
            else:
                ok = False
                if len(phrase) > 0 and phrase[0] in CHARS_TO_IGNORE:
                    phrase = phrase[1:]
                    ok = True
                if len(phrase) > 0 and phrase[-1] in CHARS_TO_IGNORE:
                    phrase = phrase[:-1]
                    ok = True
                if ok and phrase in tn_vocab:
                    spoken, _ = tn_vocab[phrase]
                    normalized_paragraph += spoken + " "
                    mask[begin:end] = [1] * (end - begin)
                    break  # exit loop after success, in favor of longer phrase

        if not mask[begin]:  # if current position was not replaced by anything just copy it to the output as is
            normalized_paragraph += pseudo_words[begin] + " "

    return normalized_paragraph


def pseudo_normalize(tn_vocab: Dict[str, Tuple[str, int]]) -> None:
    out = open(args.output_file, "w", encoding="utf-8")
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            paragraph = line.strip().casefold()  # lowercase
            paragraph = process_paragraph(paragraph, tn_vocab)
            paragraph = preprocess_apostrophes_space_diacritics(paragraph)
            paragraph = re.sub(r"(\d)\-(\d)", "\g<1> \g<2>", paragraph)  # delete hyphen between digits
            paragraph = re.sub(r" \- ", " ", paragraph)  # delete hyphen between spaces
            # replace oov characters with space
            paragraph = re.sub(r"[^ '\-aiuenrbomkygwthszdcjfvplxq1234567890]", " ", paragraph)
            paragraph = " ".join(paragraph.split())
            paragraph = process_paragraph(paragraph, tn_vocab)  # try to normalize once more on cleaned text
            out.write(paragraph + "\n")

    out.close()


def main() -> None:
    tn_vocab = {}
    with open(args.tn_vocab, "r", encoding="utf-8") as f:
        for line in f:
            sem, spoken, written, freq = line.strip().split("\t")
            if sem in {"ELECTRONIC", "LETTERS", "PLAIN", "PUNCT", "VERBATIM"}:
                continue
            if not re.search(r'\d', written):  # skip phrases without digit
                continue
            freq = int(freq)
            best_freq = 0
            if written in tn_vocab:
                _, best_freq = tn_vocab[written]
            if freq > best_freq:
                tn_vocab[written] = (spoken, freq)

    print(len(tn_vocab))
    pseudo_normalize(tn_vocab)


if __name__ == "__main__":
    main()
