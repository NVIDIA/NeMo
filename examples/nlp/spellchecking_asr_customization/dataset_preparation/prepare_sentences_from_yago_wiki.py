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
This script is used to extract sentences from Yago Wikipedia articles.
"""

import argparse
import os
import re
import tarfile
from typing import Dict, Set

from nemo.collections.nlp.data.spellchecking_asr_customization.utils import (
    get_paragraphs_from_text,
    get_title_and_text_from_json,
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
parser.add_argument(
    "--sub_misspells_file", required=True, type=str, help="File with subphrase misspells from YAGO entities"
)
parser.add_argument("--idf_file", required=True, type=str, help="File with idf of YAGO entities and their subphrases")

args = parser.parse_args()


def extract_sentences(
    input_folder: str,
    output_file: str,
    exclude_titles: Set[str],
    yago_entities: Set[str],
    sub_yago_entities: Set[str],
    idf: Dict[str, float],
) -> None:
    out = open(output_file, "w", encoding="utf-8")
    for name in os.listdir(input_folder):
        print(name)
        tar = tarfile.open(os.path.join(input_folder, name), "r:gz")
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is None:
                continue
            byte = f.read()
            f.close()
            content = byte.decode("utf-8")
            text, title, title_clean = get_title_and_text_from_json(content, exclude_titles)
            if text is None or title is None:
                continue
            title_clean_spaced = " " + title_clean + " "
            for p, p_clean in get_paragraphs_from_text(text):
                words = p_clean.split()
                phrases = set()
                sub_phrases = set()
                for begin in range(len(words)):
                    for end in range(begin + 1, min(begin + 5, len(words) + 1)):
                        max_word_idf = 0.0  # here we store idf of rarest word in the phrase, to filter out phrases like "placed within"
                        for w in words[begin:end]:
                            if w in idf:
                                if idf[w] > max_word_idf:
                                    max_word_idf = idf[w]
                            else:
                                max_word_idf = 100.0
                        phrase = " ".join(words[begin:end])
                        if phrase == title_clean:  # whole phrase is yago_entity and equals title (no filtering)
                            phrases.add(phrase)
                            continue
                        if max_word_idf < 5.0:
                            continue
                        if phrase in yago_entities:
                            phrases.add(phrase)
                        elif phrase in sub_yago_entities:
                            if max_word_idf > 7.0:
                                sub_phrases.add(phrase)
                p_clean_spaced = " " + p_clean + " "
                matches = []
                for phrase in phrases | sub_phrases:
                    pattern = " " + phrase + " "
                    matches += list(re.finditer(pattern, p_clean_spaced))
                final_phrases = set()
                for m in matches:
                    begin = m.start()
                    end = m.end() - 2
                    phrase_lower = p_clean[begin:end]
                    phrase_orig = p[begin:end]
                    if (
                        phrase_lower != phrase_orig
                        or phrase_lower in title_clean_spaced
                        or (phrase_lower not in idf or idf[phrase_lower] > 9.0)
                    ):
                        final_phrases.add(phrase_lower)
                if len(final_phrases) > 0:
                    out.write(";".join(list(final_phrases)) + "\t" + p + "\n")
    out.close()


# these are common words/phrases that for some reason occur sub_misspells but do not occur in yago_entities and thus have idf = +inf
EXCLUDE_PHRASES = {
    "abortion law",
    "acne",
    "act up",
    "action learning",
    "action research",
    "activated",
    "activator",
    "active directory",
    "acyclic",
    "adaptive reuse",
    "addictive",
    "additive",
    "admiration",
    "adulthood",
    "allegedly",
    "army's",
    "assaulted",
    "assert",
    "assuming",
    "atlas",
    "attend",
    "aug",
    "australians",
    "backgrounds",
    "beheaded",
    "chromosomes",
    "cited",
    "cites",
    "classmates",
    "combatant",
    "comics",
    "commandery",
    "considerable",
    "contestant",
    "contracted",
    "cups",
    "currently",
    "dec",
    "depending",
    "devastating",
    "devil",
    "disastrous",
    "documented",
    "entrepreneurship",
    "estranged",
    "evicted",
    "exceeding",
    "exists",
    "feb",
    "falsely",
    "family",
    "fantastic",
    "flags",
    "forthcoming",
    "gameplay",
    "grossing",
    "hardcover",
    "inflicted",
    "informs",
    "inhabit",
    "inn",
    "involve",
    "involved",
    "journal citation reports",
    "jul",
    "jun",
    "kilometres",
    "lab",
    "labs",
    "likewise",
    "listened",
    "listeners",
    "mad",
    "mar",
    "medalist",
    "mentions",
    "minds",
    "mon",
    "mutually",
    "needing",
    "neighbouring",
    "noted",
    "noting",
    "notoriety",
    "nouns",
    "nov",
    "obesity",
    "oblast",
    "obliged",
    "occur",
    "oct",
    "opposes",
    "papa",
    "prevents",
    "proclaimed",
    "promoted",
    "proposes",
    "provided",
    "pseudonym",
    "recipient",
    "recruits",
    "relegation",
    "remix",
    "reparation",
    "replaced",
    "resigned",
    "reuse",
    "revealing",
    "reverted",
    "rewarded",
    "school's",
    "scored",
    "sept",
    "sharing economy",
    "shortly",
    "sounded",
    "spared",
    "synonym",
    "synthesized",
    "tactical",
    "tel",
    "the germans",
    "the mask",
    "thorough",
    "toured",
    "translations",
    "tuesday",
    "ultra",
    "umm",
    "undrafted",
    "updated",
    "upgraded",
    "zombies",
    "vii",
    "visited",
    "wednesday",
    "worlds",
}


if __name__ == "__main__":
    n = 0
    exclude_titles = set()
    with open(args.exclude_titles_file, "r", encoding="utf-8") as f:
        for line in f:
            exclude_titles.add(line.strip())

    yago_entities = load_yago_entities(args.yago_entities_file, exclude_titles)

    idf = {}
    with open(args.idf_file, "r", encoding="utf-8") as f:
        for line in f:
            phrase, score, freq = line.strip().split("\t")
            score = float(score)
            if score > 10.0:  # phrases with score above this will never be filtered based on idf
                break
            idf[phrase] = score

    for phrase in EXCLUDE_PHRASES:
        yago_entities.discard(phrase)

    sub_yago_entities = set()
    with open(args.sub_misspells_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            phrase = parts[0]
            if phrase in EXCLUDE_PHRASES:
                continue
            words = phrase.split()
            # skip if phrase starts or ends with a very frequent word
            if words[0] in idf and idf[words[0]] < 2.5:
                continue
            if words[-1] in idf and idf[words[-1]] < 2.5:
                continue
            sub_yago_entities.add(phrase)

    extract_sentences(args.input_folder, args.output_file, exclude_titles, yago_entities, sub_yago_entities, idf)
