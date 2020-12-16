# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import sys
from typing import List

from tools.text_normalization.tagger import (
    Tag,
    tag_cardinal,
    tag_date,
    tag_decimal,
    tag_measure,
    tag_money,
    tag_ordinal,
    tag_time,
    tag_verbatim,
    tag_whitelist,
)
from tqdm import tqdm

taggers = [
    tag_whitelist,
    tag_money,
    tag_measure,
    tag_time,
    tag_decimal,
    tag_date,
    tag_ordinal,
    tag_cardinal,
    tag_verbatim,
]


def find_tags(text: str) -> List[Tag]:
    """
    Given text use all taggers to find all possible tags within the text 
    Args:
        text: string
    Returns: List of tags
    """
    tags = []
    for tagger in taggers:
        tags.extend(tagger(text))
    return tags


def select_tags(tags: List[Tag]) -> List[Tag]:
    """
    from all possible given tags to a given text select only those that are non-overlapping.
    This can have multiple strategies.
    Args:
        tags: list of tags
    Returns: List of tags that are not overlapping. Should be subset of input list of tags
    """
    res = []
    for tag in tags:
        overlapping = False
        for existing in res:
            if Tag.overlap(existing, tag):
                overlapping = True
                break
        if not overlapping:
            res.append(tag)
    return res


def verbalizer(text: str, tags: List[Tag]) -> str:
    """
    Given text and corresponding list of tags. Applies verbalizations for tagged substrings and return transduced text.
    This is context-independent, i.e. normalization only looks at tagged substring.
    Args:
        text: input text
        tags: list of tags of input text
    Returns: normalized input text
    """
    # sort by last starting point
    tags = sorted(tags, key=lambda x: -x.start)
    for tag in tags:
        text = text[: tag.start] + tag.normalize(tag.data) + text[tag.end :]
    return text


def normalize_numbers(text: str, verbose: bool):
    """
    main function. normalizes alphanumerical tokens in given text to its verbalized form:
    e.g. "12kg -> twelve kilograms"
    Args:
        text: string that may include semiotic classes.
    Returns: verbalized form in string format
    """
    tags = find_tags(text)
    tags = select_tags(tags)
    output = verbalizer(text, tags)
    if verbose:
        print([str(tag) for tag in tags])
        print(output)
    return output


def normalize_identity(un_normalized: List[str], verbose: bool = False) -> List[str]:
    """
    Identity normalizer. Returns input unchanged
    Args:
        un_normalized: input string
    Returns input string
    """
    return un_normalized


def normalize_nemo(un_normalized: List[str], verbose: bool = False) -> List[str]:
    """
    Text normalization with NeMo algorithm.
    Args:
        un_normalized: List of unnormlized strings
        verbose: if specified prints debugging info
    Returns list of normalized strings
    """
    res = []
    for input in tqdm(un_normalized):
        text = normalize_numbers(input, verbose=verbose)
        res.append(text)
    return res


normalizers = {"identity": normalize_identity, "nemo": normalize_nemo}


if __name__ == "__main__":
    # Example usage:
    s = sys.argv[1]  # input string
    normalize_numbers(s, verbose=True)
