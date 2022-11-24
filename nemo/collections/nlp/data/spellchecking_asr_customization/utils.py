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


import json
import re
from typing import Dict, List, Set, Tuple

"""Utility functions for Spellchecking ASR Customization."""



SPACE_REGEX = re.compile(r"[\u2000-\u200F]", re.UNICODE)
APOSTROPHES_REGEX = re.compile(r"[’'‘`ʽ']")
CHARS_TO_IGNORE_REGEX = re.compile(r"[\.\,\?\:\-!;()«»…\]\[/\*–‽+&_\\½√>€™$•¼}{~—=“\"”″‟„]")


def replace_diacritics(text):
    text = re.sub(r"[éèëēêęěė]", "e", text)
    text = re.sub(r"[ãâāáäăâàąåạả]", "a", text)
    text = re.sub(r"[úūüùưûů]", "u", text)
    text = re.sub(r"[ôōóöõòő]", "o", text)
    text = re.sub(r"[ćçč]", "c", text)
    text = re.sub(r"[ïīíîıì]", "i", text)
    text = re.sub(r"[ñńňņ]", "n", text)
    text = re.sub(r"[țť]", "t", text)
    text = re.sub(r"[łľ]", "l", text)
    text = re.sub(r"[żžź]", "z", text)
    text = re.sub(r"[ğ]", "g", text)
    text = re.sub(r"[ř]", "r", text)
    text = re.sub(r"[ý]", "y", text)
    text = re.sub(r"[æ]", "ae", text)
    text = re.sub(r"[œ]", "oe", text)
    text = re.sub(r"[șşšś]", "s", text)
    return text


def preprocess_apostrophes_space_diacritics(text):
    text = APOSTROPHES_REGEX.sub("'", text) # replace different apostrophes by one
    text = re.sub(r"'+", "'", text)  # merge multiple apostrophes
    text = SPACE_REGEX.sub(" ", text) # replace different spaces by one
    text = replace_diacritics(text)

    text = re.sub(r" '", " ", text)  # delete apostrophes at the beginning of word
    text = re.sub(r"' ", " ", text)  # delete apostrophes at the end of word
    text = re.sub(r" +", " ", text)  # merge multiple spaces
    return text


def get_title_and_text_from_json(content: str, exclude_titles: Set[str]) -> Tuple[str, str, str]:
    # Example of file content
    #   {"query":
    #     {"normalized":[{"from":"O'_Coffee_Club","to":"O' Coffee Club"}],
    #      "pages":
    #       {"49174116":
    #         {"pageid":49174116,
    #          "ns":0,
    #          "title":"O' Coffee Club",
    #          "extract":"O' Coffee Club (commonly known as Coffee Club) is a Singaporean coffee house..."
    #         }
    #       }
    #     }
    #   }
    try:
        js = json.loads(content.strip())
    except:
        print("cannot load json from text")
        return (None, None, None)
    if "query" not in js or "pages" not in js["query"]:
        print("no query[\"pages\"] in " + content)
        return (None, None, None)
    for page_key in js["query"]["pages"]:
        if page_key == "-1":
            continue
        page = js["query"]["pages"][page_key]
        if "title" not in page:
            continue
        title = page["title"]
        if title in exclude_titles:
            return (None, None, None)
        if "extract" not in page:
            continue
        text = page["extract"]
        title_clean = preprocess_apostrophes_space_diacritics(title)
        title_clean = CHARS_TO_IGNORE_REGEX.sub(" ", title_clean).lower()  # number of characters is the same in p and p_clean 
        return text, title, title_clean
    return (None, None, None)


def get_paragraphs_from_text(text):
    paragraphs = text.split("\n")
    for paragraph in paragraphs:
        if paragraph == "":
            continue
        p = preprocess_apostrophes_space_diacritics(paragraph)
        p_clean = CHARS_TO_IGNORE_REGEX.sub(" ", p).lower()  # number of characters is the same in p and p_clean 
        yield p, p_clean


def get_paragraphs_from_json(text, exclude_titles):
    # Example of file content
    #   {"query":
    #     {"normalized":[{"from":"O'_Coffee_Club","to":"O' Coffee Club"}],
    #      "pages":
    #       {"49174116":
    #         {"pageid":49174116,
    #          "ns":0,
    #          "title":"O' Coffee Club",
    #          "extract":"O' Coffee Club (commonly known as Coffee Club) is a Singaporean coffee house..."
    #         }
    #       }
    #     }
    #   }
    try:
        js = json.loads(text.strip())
    except:
        print("cannot load json from text")
        return
    if "query" not in js or "pages" not in js["query"]:
        print("no query[\"pages\"] in " + text)
        return
    for page_key in js["query"]["pages"]:
        if page_key == "-1":
            continue
        page = js["query"]["pages"][page_key]
        if "title" not in page:
            continue
        title = page["title"]
        if title in exclude_titles:
            continue
        if "extract" not in page:
            continue
        text = page["extract"]
        paragraphs = text.split("\n")
        for paragraph in paragraphs:
            if paragraph == "":
                continue
            p = preprocess_apostrophes_space_diacritics(paragraph)
            p_clean = CHARS_TO_IGNORE_REGEX.sub(" ", p).lower()  # number of characters is the same in p and p_clean 
            yield p, p_clean


def load_yago_entities(input_name: str, exclude_titles: Set[str]) -> Set[str]:
    yago_entities = set()
    with open(input_name, "r", encoding="utf-8") as f:
        for line in f:
            title_orig, title_clean = line.strip().split("\t")
            title_clean = title_clean.replace("_", " ")
            title_orig = title_orig.replace("_", " ")
            if title_orig in exclude_titles:
                print("skip: ", title_orig)
                continue
            yago_entities.add(title_clean)
    return yago_entities


def get_token_list(text: str) -> List[str]:
    """Returns a list of tokens.

    This function expects that the tokens in the text are separated by space
    character(s). Example: "ca n't , touch". This is the case at least for the
    public DiscoFuse and WikiSplit datasets.

    Args:
        text: String to be split into tokens.
    """
    return text.split()


def read_label_map(path: str) -> Dict[str, int]:
    """Return label map read from the given path."""
    with open(path, 'r') as f:
        label_map = {}
        empty_line_encountered = False
        for tag in f:
            tag = tag.strip()
            if tag:
                label_map[tag] = len(label_map)
            else:
                if empty_line_encountered:
                    raise ValueError('There should be no empty lines in the middle of the label map ' 'file.')
                empty_line_encountered = True
        return label_map
