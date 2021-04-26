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

from nemo_text_processing.inverse_text_normalization.graph_utils import GraphFst, delete_extra_space, delete_space
from nemo_text_processing.inverse_text_normalization.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.taggers.date import DateFst
from nemo_text_processing.inverse_text_normalization.taggers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.taggers.measure import MeasureFst
from nemo_text_processing.inverse_text_normalization.taggers.money import MoneyFst
from nemo_text_processing.inverse_text_normalization.taggers.ordinal import OrdinalFst
from nemo_text_processing.inverse_text_normalization.taggers.punctuation import PunctuationFst
from nemo_text_processing.inverse_text_normalization.taggers.time import TimeFst
from nemo_text_processing.inverse_text_normalization.taggers.whitelist import WhiteListFst
from nemo_text_processing.inverse_text_normalization.taggers.word import WordFst

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class ClassifyFst(GraphFst):
    """
    Composes other classfier grammars. This class will be compiled and exported to thrax FAR. 
    """

    def __init__(self):
        super().__init__(name="tokenize_and_classify", kind="classify")

        cardinal_graph_fst = CardinalFst()
        cardinal = cardinal_graph_fst.fst

        ordinal_graph_fst = OrdinalFst(cardinal_graph_fst)
        ordinal = ordinal_graph_fst.fst

        decimal_graph_fst = DecimalFst(cardinal_graph_fst)
        decimal = decimal_graph_fst.fst

        measure = MeasureFst(cardinal_graph_fst, decimal_graph_fst).fst
        date = DateFst(ordinal_graph_fst).fst
        word = WordFst().fst
        time = TimeFst().fst
        money = MoneyFst(cardinal_graph_fst, decimal_graph_fst).fst
        whitelist = WhiteListFst().fst

        classify = (
            pynutil.add_weight(whitelist, 1.01)
            | pynutil.add_weight(time, 1.1)
            | pynutil.add_weight(date, 1.09)
            | pynutil.add_weight(decimal, 1.1)
            | pynutil.add_weight(measure, 1.1)
            | pynutil.add_weight(cardinal, 1.1)
            | pynutil.add_weight(ordinal, 1.1)
            | pynutil.add_weight(money, 1.1)
            | pynutil.add_weight(word, 100)
        )

        punct = (
            pynutil.insert("tokens { ") + pynutil.add_weight(PunctuationFst().fst, weight=1.1) + pynutil.insert(" }")
        )
        token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
        token_plus_punct = (
            pynini.closure(punct + pynutil.insert(" ")) + token + pynini.closure(pynutil.insert(" ") + punct)
        )

        graph = token_plus_punct + pynini.closure(delete_extra_space + token_plus_punct)
        graph = delete_space + graph + delete_space

        self.fst = graph.optimize()
