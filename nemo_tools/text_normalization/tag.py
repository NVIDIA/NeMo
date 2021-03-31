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
from enum import Enum


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
        data: ordered dict of class dependent meta info
        weight: weight of tag within unnormalized text, default 1.0
        text: semiotic token string
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
        Args:
            tag1: first tag object
            tag2: second tag object
        Returns true if both tags overlap in their respective span
        """
        return (tag1.start <= tag2.start < tag1.end) or (tag2.start <= tag1.start < tag2.end)

    def __str__(self):
        return json.dumps(self.__dict__, cls=EnumEncoder)
