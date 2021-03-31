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

from nemo_tools.text_normalization.tag import Tag, TagType
from nemo_tools.text_normalization.tagger import (
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
from nemo_tools.text_normalization.verbalizer import (
    expand_cardinal,
    expand_date,
    expand_decimal,
    expand_digit,
    expand_electronic,
    expand_fraction,
    expand_letter,
    expand_measurement,
    expand_money,
    expand_ordinal,
    expand_punct,
    expand_roman,
    expand_telephone,
    expand_time,
    expand_verbatim,
    expand_whitelist,
)
from tqdm import tqdm

TAGGERS = [
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

VERBALIZERS = {
    TagType.CARDINAL: [expand_cardinal, expand_roman],
    TagType.DATE: [expand_date],
    TagType.DECIMAL: [expand_decimal],
    TagType.DIGIT: [expand_digit],
    TagType.ELECTRONIC: [expand_electronic],
    TagType.FRACTION: [expand_fraction],
    TagType.LETTERS: [expand_letter],
    TagType.MEASURE: [expand_measurement],
    TagType.MONEY: [expand_money],
    TagType.ORDINAL: [expand_ordinal],
    TagType.PUNCT: [expand_punct],
    TagType.TELEPHONE: [expand_telephone],
    TagType.TIME: [expand_time],
    TagType.VERBATIM: [expand_verbatim],
    TagType.WHITELIST: [expand_whitelist],
}


def find_tags(text: str) -> List[Tag]:
    """
    Given text use all taggers to find all possible tags within the text 
    Args:
        text: string
    Returns: List of tags
    """
    tags = []
    for tagger in TAGGERS:
        foundTags = find_tag(text, tagger)
        if foundTags:
            tags.extend(foundTags)
    return tags


def find_tag(text: str, tagger) -> List[Tag]:
    """
    Given text and tagger find all matching tags
    Args:
        text: string
        tagger: tagger
    Returns: List of Tags or None
    """
    return tagger(text)


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


def verbalize(text: str, tags: List[Tag]) -> str:
    """
    Given text and corresponding list of tags. Applies verbalization where possible for tagged substrings and returns transduced text.
    This is context-independent, i.e. normalization only looks at tagged substring.
    Args:
        text: input text
        tags: list of tags of input text
    Returns: normalized input text
    """
    # sort by last starting point
    tags = sorted(tags, key=lambda x: -x.start)
    for tag in tags:
        text = text[: tag.start] + _verbalize(tag) + text[tag.end :]
    return text


def _verbalize(tag: Tag) -> str:
    """
    Given tag applies verbalization if possible and returns transduced text.
    This is context-independent.
    Args:
        tag: tag
    Returns: verbalized text
    """
    expand_funcs = VERBALIZERS[tag.kind]
    res = [f(tag.data) for f in expand_funcs]
    res = [x for x in res if x]
    if not res:
        return tag.text
    else:
        return res[0]


def normalize(text: str, verbose: bool):
    """
    main function. normalizes alphanumerical tokens in given text to its verbalized form:
    e.g. "12kg -> twelve kilograms"
    Args:
        text: string that may include semiotic classes.
    Returns: verbalized form in string format
    """
    tags = find_tags(text)
    tags = select_tags(tags)
    output = verbalize(text, tags)
    if verbose:
        print(text)
        print(output)
        print([str(tag) for tag in tags])
    return output


def normalize_identity(texts: List[str], verbose: bool = False) -> List[str]:
    """
    Identity normalizer. Returns input unchanged
    Args:
        texts: input string
    Returns input string
    """
    return texts


def normalize_nemo(texts: List[str], verbose: bool = False) -> List[str]:
    """
    Text normalization with NeMo algorithm.
    Args:
        texts: List of unnormlized strings
        verbose: if specified prints debugging info
    Returns list of normalized strings
    """
    res = []
    for input in tqdm(texts):
        text = normalize(input, verbose=verbose)
        res.append(text)
    return res


normalizers = {"identity": normalize_identity, "nemo": normalize_nemo}


if __name__ == "__main__":
    # Example usage:
    s = sys.argv[1]  # input string
    normalize_numbers(s, verbose=True)
