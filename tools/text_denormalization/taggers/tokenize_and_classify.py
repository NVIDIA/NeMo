# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from denormalization.graph_utils import GraphFst
from denormalization.taggers.cardinal import CardinalFst
from denormalization.taggers.date import DateFst
from denormalization.taggers.decimal import DecimalFst
from denormalization.taggers.measure import MeasureFst
from denormalization.taggers.money import MoneyFst
from denormalization.taggers.ordinal import OrdinalFst
from denormalization.taggers.punctuation import PunctuationFst
from denormalization.taggers.time import TimeFst
from denormalization.taggers.whitelist import WhiteListFst
from denormalization.taggers.word import WordFst
from pynini.lib import pynutil


class ClassifyFst(GraphFst):
    def __init__(self):
        super().__init__(name="tokenize_and_classify", kind="classify")

        cardinal = CardinalFst().fst
        ordinal = OrdinalFst().fst
        decimal = DecimalFst().fst
        measure = MeasureFst().fst
        date = DateFst().fst
        word = WordFst().fst
        punct = PunctuationFst().fst
        time = TimeFst().fst
        money = MoneyFst().fst
        whitelist = WhiteListFst().fst
        types = whitelist | date | ordinal | decimal | measure | cardinal | time | money | word
        token = pynutil.insert("tokens { ") + types + pynutil.insert(" }")
        token_plus_punct = (
            pynini.closure(punct + pynutil.insert(" ")) + token + pynini.closure(pynutil.insert(" ") + punct)
        )
        graph = token_plus_punct + pynini.closure(pynini.cross(pynini.closure(" ", 1), " ") + token_plus_punct)
        graph = (
            pynini.closure(pynutil.delete(pynini.closure(" ", 1)), 0, 1)
            + graph
            + pynini.closure(pynutil.delete(pynini.closure(" ", 1)), 0, 1)
        )
        self.fst = graph.optimize()
