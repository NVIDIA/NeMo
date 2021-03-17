# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

import pynini
from pynini.lib import pynutil
from ..graph_utils import GraphFst, delete_extra_space, delete_space
from .cardinal import CardinalFst
from .date import DateFst
from .decimal import DecimalFst
from .measure import MeasureFst
from .money import MoneyFst
from .ordinal import OrdinalFst
from .punctuation import PunctuationFst
from .time import TimeFst
from .word import WordFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars. This class will be compiled and exported to thrax FAR.
    """

    def __init__(self):
        super().__init__(name="verbalize", kind="verbalize")
        cardinal = CardinalFst().fst
        ordinal = OrdinalFst().fst
        decimal = DecimalFst().fst
        measure = MeasureFst().fst
        punct = PunctuationFst().fst
        time = TimeFst().fst
        word = WordFst().fst
        date = DateFst().fst
        money = MoneyFst().fst
        types = date | money | time | measure | ordinal | decimal | cardinal | word | punct
        graph = (
            pynutil.delete("tokens", weight=-10)
            + delete_space
            + pynutil.delete("{")
            + delete_space
            + types
            + delete_space
            + pynutil.delete("}")
        )
        graph = delete_space + pynini.closure(graph + delete_extra_space) + graph + delete_space
        self.fst = graph
