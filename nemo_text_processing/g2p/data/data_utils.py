# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


import csv
import re
import string
from typing import List

__all__ = ['correct_wikihomograph_data', 'read_wikihomograph_file', 'read_wordids']


def correct_wikihomograph_data(sentence: str, start: int = None, end: int = None):
    """
    Correct indices for WikiHomograph data

    Args:
        sentence: sentence
        start: start index of homograph
        end: end index of homograph

    """
    corrections = {
        "It is traditionally composed of 85–99% tin, mixed with copper, antimony, bismuth, and sometimes lead, although the use of lead is less common today.": [
            96,
            100,
        ],
        "B₁₀₅ can be conceptually divided into a B₄₈ fragment and B₂₈-B-B₂₈ (B₅₇) fragment.": [44, 52],
        "Pierrefonds Airport on Réunion recorded just 18 mm (0.71 in) of rainfall from November to January, a record minimum.": [
            101,
            107,
        ],
        "Consort Chen Farong (陳法容) was an imperial consort during the Chinese dynasty Liu Song.": [42, 49],
        "Unlike TiO₂, which features six-coordinate Ti in all phases, monoclinic zirconia consists of seven-coordinate zirconium centres.": [
            32,
            42,
        ],
        "Its area is 16 km², its approximate length is 10 km, and its approximate width is 3 km.": [24, 35],
        "The conjugate momentum to X has the expressionwhere the pᵢ are the momentum functions conjugate to the coordinates.": [
            86,
            95,
        ],
        "Furthermore 17β-HSD1 levels positively correlate with E2 and negatively correlate with DHT levels in breast cancer cells.": [
            39,
            48,
        ],
        "Electric car buyers get a €4,000 (US$4,520) discount while buyers of plug-in hybrid vehicles get a discount of €3,000 (US$3,390).": [
            99,
            107,
        ],
    }

    if sentence in corrections:
        start, end = corrections[sentence]

    sentence = sentence.replace("2014Coordinate", "2014 Coordinate")  # for normalized data for G2P OOV models
    sentence = sentence.replace("AAA", "triple A")  # for normalized data for G2P OOV models

    return sentence, start, end


def read_wikihomograph_file(file: str) -> (List[str], List[List[int]], List[str], List[str]):
    """
    Reads .tsv file from WikiHomograph dataset,
    e.g. https://github.com/google-research-datasets/WikipediaHomographData/blob/master/data/eval/live.tsv

    Args:
        file: path to .tsv file
    Returns:
        sentences: Text.
        start_end_indices: Start and end indices of the homograph in the sentence.
        homographs: Target homographs for each sentence
        word_ids: Word_ids corresponding to each homograph, i.e. label.
    """
    excluded_sentences = 0
    num_corrected = 0
    sentences = []
    start_end_indices = []
    homographs = []
    word_ids = []
    with open(file, "r", encoding="utf-8") as f:
        tsv_file = csv.reader(f, delimiter="\t")
        for i, line in enumerate(tsv_file):
            if i == 0:
                continue
            homograph, wordid, sentence, start, end = line
            start, end = int(start), int(end)
            sentence, start, end = correct_wikihomograph_data(sentence, start, end)

            homograph_span = sentence[start:end]
            if homograph_span.lower() != homograph:
                if sentence.lower().count(homograph) == 1:
                    start = sentence.lower().index(homograph)
                    end = start + len(homograph)
                    homograph_span = sentence[start:end].lower()
                    assert homograph == homograph_span.lower()
                else:
                    excluded_sentences += 1
                    raise ValueError(f"homograph {homograph} != homograph_span {homograph_span} in {sentence}")

            homographs.append(homograph)
            start_end_indices.append([start, end])
            sentences.append(sentence)
            word_ids.append(wordid)

    if num_corrected > 0:
        print(f"corrected: {num_corrected}")
    if excluded_sentences > 0:
        print(f"excluded: {excluded_sentences}")
    return sentences, start_end_indices, homographs, word_ids


def read_wordids(wordid_map):
    """
    Reads wordid file from WikiHomograph dataset,
    https://github.com/google-research-datasets/WikipediaHomographData/blob/master/data/wordids.tsv

    Args:
        wordid_map: path to wordids.tsv
    Returns:
        homograph_dict: a dictionary of graphemes with corresponding word_id - ipa_form pairs
        wordid_to_idx: word id to label id mapping
    """
    homograph_dict = {}
    wordid_to_idx = {}

    with open(wordid_map, "r", encoding="utf-8") as f:
        tsv_file = csv.reader(f, delimiter="\t")

        for i, line in enumerate(tsv_file):
            if i == 0:
                continue

            grapheme = line[0]
            word_id = line[1]
            ipa_form = line[3]
            wordid_to_idx[word_id] = len(wordid_to_idx)
            if grapheme not in homograph_dict:
                homograph_dict[grapheme] = {}
            homograph_dict[grapheme][word_id] = ipa_form
    return homograph_dict, wordid_to_idx


def get_wordid_to_nemo():
    wordid_to_nemo_cmu = {}
    with open("/home/ebakhturina/g2p_scripts/misc_data/wordid_to_nemo_cmu.tsv", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            line = line.strip().split("\t")
            wordid_to_nemo_cmu[line[0]] = line[1]
    return wordid_to_nemo_cmu


def remove_punctuation(text: str, exclude=None):
    all_punct_marks = string.punctuation

    if exclude is not None:
        for p in exclude:
            all_punct_marks = all_punct_marks.replace(p, "")
    text = re.sub("[" + all_punct_marks + "]", " ", text)

    text = re.sub(r" +", " ", text)
    return text.strip()
