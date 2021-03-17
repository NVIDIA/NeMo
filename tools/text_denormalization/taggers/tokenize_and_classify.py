# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from pynini.lib import pynutil
from ..graph_utils import GraphFst
from .cardinal import CardinalFst
from .date import DateFst
from .decimal import DecimalFst
from .measure import MeasureFst
from .money import MoneyFst
from .ordinal import OrdinalFst
from .time import TimeFst
from .whitelist import WhiteListFst
from .word import WordFst


class ClassifyFst(GraphFst):
    """
    Composes other classfier grammars. This class will be compiled and exported to thrax FAR. 
    """

    def __init__(self):
        super().__init__(name="tokenize_and_classify", kind="classify")

        cardinal = CardinalFst().fst
        ordinal = OrdinalFst().fst
        decimal = DecimalFst().fst
        measure = MeasureFst().fst
        date = DateFst().fst
        word = WordFst().fst
        time = TimeFst().fst
        money = MoneyFst().fst
        whitelist = WhiteListFst().fst
        graph = (
            pynutil.add_weight(whitelist, 1.0)
            | pynutil.add_weight(time, 1.1)
            | pynutil.add_weight(date, 1.01)
            | pynutil.add_weight(decimal, 1.1)
            | pynutil.add_weight(measure, 1.1)
            | pynutil.add_weight(cardinal, 1.1)
            | pynutil.add_weight(ordinal, 1.1)
            | pynutil.add_weight(money, 1.1)
            | pynutil.add_weight(word, 100)
        )
        self.fst = graph.optimize()
