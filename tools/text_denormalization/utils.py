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

import json
import string
from enum import Enum
from typing import List, Union

import inflect
import regex as re

_inflect = inflect.engine()


def num_to_word(x: Union[str, int]):
    if isinstance(x, int):
        x = str(x)
    x = _inflect.number_to_words(str(x)).replace("-", " ").replace(",", "")
    return x


class TagType(Enum):
    """
    Class for Tagger types
    """

    PLAIN = 1
    PUNCT = 2
    DATE = 3
    CARDINAL = 4  # counting
    LETTERS = 5
    VERBATIM = 6
    MEASURE = 7
    DECIMAL = 8
    ORDINAL = 9
    DIGIT = 10
    MONEY = 11
    TELEPHONE = 12
    ELECTRONIC = 13
    FRACTION = 14
    TIME = 15
    ADDRESS = 16
    WHITELIST = 17


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        return str(obj)


class Tag:
    """
    Class for tagger object that is created for each detected semiotic token.
    Args:
        kind: TagType
        start: start index of unnormalized substring
        end: end index of unnormalized substring
        data: ordereddict of class dependent meta info
        text: semiotic tokens
    """

    def __init__(self, kind, start, end, data, weight=1.0):
        self.kind = kind
        self.start = start
        self.end = end
        self.data = data  # has order too
        self.weight = weight
        self.text = data["value"]

    @staticmethod
    def overlap(tag1, tag2) -> bool:
        """
        checks if given tags overlap in their text span
        """
        return (tag1.start <= tag2.start < tag1.end) or (tag2.start <= tag1.start < tag2.end)

    def __str__(self):
        return json.dumps(self.__dict__, cls=EnumEncoder)


def make_re(re_inner: str, *args):
    """
    Takes given regex expression and decorates it with left and right word boundaries
    Args:
        re_inner: regex
        args: list of optional regex arguments
    Returns compiled regex
    """
    return re.compile(rf'(?P<value>{re_inner})', *args)


def re_tag(text, kind: TagType, regex) -> List[Tag]:
    """
    Detects and returns all tags in the text.
    Args:
        text: string
        kind: Tag type
        regex: compiled regex for detection
    Returns: generates all semiotic class tags that appear in the text
    """
    for match in re.finditer(regex, text, overlapped=True):
        yield Tag(
            kind=kind, start=match.start("value"), end=match.end("value"), data=match.groupdict(),
        )


def fst_tag(text, kind) -> List[Tag]:
    # find all <value> <value> and sub <> .. </> inside </value> and create Tag from it

    r_tag_value = re.compile(rf'<(?P<tag>value)>(?P<content>.*?)</(?P=tag)>')
    r_tag_general = re.compile(rf'<(?P<tag>[a-z]*)>(?P<content>.*?)</(?P=tag)>')
    offset = 0
    for match in re.finditer(r_tag_value, text, overlapped=True):
        d = dict()
        content = match.groupdict()['content']
        tag_name = match.groupdict()['tag']
        repl = lambda m: m.groupdict()['content']
        d[tag_name] = r_tag_general.sub(repl, content)
        offset += len(tag_name) + 2
        start_index = match.start('content') - offset
        for inner_match in re.finditer(r_tag_general, content, overlapped=True):
            inner_content = inner_match.groupdict()['content']
            inner_tag_name = inner_match.groupdict()['tag']
            offset += len(inner_tag_name) + 2
            d[inner_tag_name] = inner_content
            offset += len(inner_tag_name) + 3

        end_index = match.end('content') - offset
        offset += len(tag_name) + 3

        yield Tag(
            kind=kind, start=start_index, end=end_index, data=d,
        )
