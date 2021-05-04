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

from nemo_text_processing.text_normalization.data_loader_utils import get_abs_path
from nemo_text_processing.text_normalization.graph_utils import (
    NEMO_SIGMA,
    SINGULAR_TO_PLURAL,
    GraphFst,
    convert_space,
    insert_space,
)

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, suppletive aware, e.g. 
        $12.05 -> money { currency: "dollars" integer_part: "twelve" fractional_part: "o five" }
        $1 -> money { currency: "dollar" integer_part: "one" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst):
        super().__init__(name="money", kind="classify")
        cardinal_graph = cardinal.graph
        graph_decimal_final = decimal.final_graph_wo_negative

        unit_singular = pynini.string_file(get_abs_path("data/currency.tsv"))
        unit_plural = convert_space(unit_singular @ SINGULAR_TO_PLURAL)
        unit_singular = convert_space(unit_singular)

        graph_unit_singular = pynutil.insert("currency: \"") + unit_singular + pynutil.insert("\"")
        graph_unit_plural = pynutil.insert("currency: \"") + unit_plural + pynutil.insert("\"")

        graph_integer = (
            graph_unit_plural
            + pynutil.insert(" integer_part: \"")
            + ((NEMO_SIGMA - "1") @ cardinal_graph)
            + pynutil.insert("\"")
        )
        graph_integer |= (
            graph_unit_singular + pynutil.insert(" integer_part: \"") + pynini.cross("1", "one") + pynutil.insert("\"")
        )
        graph_decimal = graph_unit_plural + insert_space + graph_decimal_final
        final_graph = graph_integer | graph_decimal
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
