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


import argparse
import os
import re
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True, type=str, help="Folder with audio and text subfolder")
parser.add_argument("--processed_folder", required=True, type=str, help="Folder with files like 1000_with_punct.txt")
parser.add_argument(
    "--min_len",
    required=True,
    type=int,
    help="Minimum number of characters in user phrase (including space), e.g. 6 symbols",
)
args = parser.parse_args()

idf = defaultdict(int)
with open(args.folder + "/vocabs/idf.txt", "r", encoding="utf-8") as f:
    for line in f:
        word, d = line.strip().split("\t")
        idf[word] = int(d)

for name in os.listdir(args.folder + "/text"):
    doc_id, ext = name.split(".")

    custom_phrases = set()

    # first take all phrases from vocabs/*.headings.txt, except for containing numbers.
    with open(args.folder + "/vocabs/" + doc_id + ".headings.txt", "r", encoding="utf-8") as f:
        for line in f:
            phrase = line.strip()
            if re.search(r"\d", phrase):
                continue
            phrase = re.sub(r"[^\w ']", r" ", phrase)
            phrase = " ".join(phrase.strip().split())  # delete extra spaces
            custom_phrases.add(phrase)

    # next take capitalized sequences and non-capitalized single-word terms.
    with open(args.processed_folder + "/" + doc_id + "_with_punct.txt", "r", encoding="utf-8") as f:
        for line in f:
            sentence = line.strip()
            sentence = re.sub(r"[^\w ']", r" ", sentence)
            sentence = " ".join(sentence.strip().split())  # delete extra spaces
            words = sentence.split()
            phrase_words = []
            for idx, w in enumerate(words):
                w_lower = w.casefold()
                if w_lower not in idf:
                    if len(phrase_words) > 0:
                        custom_phrases.add(" ".join(phrase_words))
                    phrase_words = []
                    continue
                if re.search(r"\d", w):
                    if len(phrase_words) > 0:
                        custom_phrases.add(" ".join(phrase_words))
                    phrase_words = []
                    continue
                if re.search(r"_", w):
                    if len(phrase_words) > 0:
                        custom_phrases.add(" ".join(phrase_words))
                    phrase_words = []
                    continue
                # if idf[w_lower] <= 2 and w_lower == w and len(w) >= 4:   # lower-cased term
                #    if len(phrase_words) > 0:
                #        custom_phrases.add(" ".join(phrase_words))
                #    custom_phrases.add(w)
                #    phrase_words = []
                if idf[w_lower] < 500 and w_lower != w and idx != 0:
                    phrase_words.append(w)
                else:
                    if len(phrase_words) > 0:
                        custom_phrases.add(" ".join(phrase_words))
                    phrase_words = []

            if len(phrase_words) > 0:
                custom_phrases.add(" ".join(phrase_words))

    out = open(args.folder + "/vocabs/" + doc_id + ".custom.txt", "w", encoding="utf-8")
    for phrase in custom_phrases:
        if len(phrase) >= args.min_len:
            out.write(phrase + "\n")
    out.close()
