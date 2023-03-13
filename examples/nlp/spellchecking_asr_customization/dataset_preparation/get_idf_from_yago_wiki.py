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
This script is used to calculate idf for words and short phrases from Yago Wikipedia articles.
"""

import argparse
import math
import os
import tarfile
from collections import defaultdict
from typing import Set

from nemo.collections.nlp.data.spellchecking_asr_customization.utils import (
    get_paragraphs_from_json,
    load_yago_entities,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_folder",
    required=True,
    type=str,
    help="Input folder with tar.gz files each containing wikipedia articles in json format",
)
parser.add_argument("--exclude_titles_file", required=True, type=str, help="File with article titles to be skipped")
parser.add_argument("--output_file", required=True, type=str, help="Output file")
parser.add_argument("--yago_entities_file", required=True, type=str, help="File with preprocessed YAGO entities")
args = parser.parse_args()


def get_idf(input_folder: str, exclude_titles: Set[str], yago_entities: Set[str]):
    """
    Args:
        input_folder: Input folder with tar.gz files each containing wikipedia articles in json format
        exclude_titles: Set of titles that should be skipped (e.g. test articles)
        yago_entities: Set of phrases that we want to find in texts

    Returns:
        idf: a dictionary where the key is a phrase, value is its inverse document frequency
    """

    n_documents = 0
    idf = defaultdict(int)
    for name in os.listdir(input_folder):
        print(name)
        tar = tarfile.open(os.path.join(input_folder, name), "r:gz")
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is None:
                continue
            byte = f.read()
            text = byte.decode("utf-8")
            n_documents += 1
            phrases = set()
            for p, p_clean in get_paragraphs_from_json(text, exclude_titles):
                words = p_clean.split()
                for begin in range(len(words)):
                    for end in range(begin + 1, min(begin + 5, len(words) + 1)):
                        phrase = " ".join(words[begin:end])
                        if phrase in yago_entities:
                            phrases.add(phrase)
            for phrase in phrases:
                idf[phrase] += 1  # one count per document
        # delete too rare phrases (no need to store their idf)
        for phrase in list(idf.keys()):
            if idf[phrase] < 4:
                del idf[phrase]

    return idf, n_documents


if __name__ == "__main__":
    n = 0
    exclude_titles = set()
    with open(args.exclude_titles_file, "r", encoding="utf-8") as f:
        for line in f:
            exclude_titles.add(line.strip())

    yago_entities = load_yago_entities(args.yago_entities_file, exclude_titles)
    # add ngrams from yago_entities as separate phrases (may need them later, need to know their idf too)
    yago_entities_plus = set()
    for phrase in yago_entities:
        words = phrase.split()
        for begin in range(len(words)):
            for end in range(begin + 1, min(begin + 5, len(words) + 1)):
                new_phrase = " ".join(words[begin:end])
                yago_entities_plus.add(new_phrase)

    idf, n_documents = get_idf(args.input_folder, exclude_titles, yago_entities.union(yago_entities_plus))

    with open(args.output_file, "w", encoding="utf-8") as out:
        for phrase, freq in sorted(idf.items(), key=lambda item: item[1], reverse=True):
            score = math.log(n_documents / freq)
            out.write(phrase + "\t" + str(score) + "\t" + str(freq) + "\n")
