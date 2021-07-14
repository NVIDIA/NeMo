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

from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.de.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
)

try:
    import pynini
    from pynini.lib import pynutil

    graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv")).optimize()
    graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).optimize()
    ties_graph = pynini.string_file(get_abs_path("data/numbers/ties.tsv")).optimize()
    zero_graph = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).optimize()

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    graph_teen = None
    graph_digit = None
    ties_graph = None

    PYNINI_AVAILABLE = True


def _get_month_graph():
    """
    Transducer for month, e.g. march -> march
    """
    month_graph = pynini.string_file(get_abs_path("data/months.tsv"))
    return month_graph


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, 
        e.g. january fifth twenty twelve -> date { month: "january" day: "5" year: "2012" preserve_order: true }
        e.g. the fifth of january twenty twelve -> date { day: "5" month: "january" year: "2012" preserve_order: true }
        e.g. twenty twenty -> date { year: "2012" preserve_order: true }

    Args:
        ordinal: OrdinalFst
    """

    def __init__(self, ordinal: GraphFst, cardinal: GraphFst):
        super().__init__(name="date", kind="classify")

        self.cardinal = cardinal
        ordinal_graph = ordinal.graph
        year_graph = self._get_year_graph()
        YEAR_WEIGHT = 0.001
        year_graph = pynutil.add_weight(year_graph, YEAR_WEIGHT)
        month_graph = _get_month_graph()

        month_graph = pynutil.insert("month: \"") + month_graph + pynutil.insert("\"")

        day_graph = pynutil.insert("day: \"") + pynutil.add_weight(ordinal_graph, -0.7) + pynutil.insert("\"")
        optional_day_graph = pynini.closure(delete_extra_space + day_graph, 0, 1)
        optional_graph_year = pynini.closure(
            delete_extra_space
            + pynutil.insert("year: \"")
            + pynutil.add_weight(year_graph, -YEAR_WEIGHT)
            + pynutil.insert("\""),
            0,
            1,
        )
        graph_dmy = day_graph + delete_extra_space + month_graph + optional_graph_year
        # graph_year = pynutil.insert("year: \"") + (year_graph | _get_range_graph()) + pynutil.insert("\"")
        graph_year = pynutil.insert("year: \"") + (year_graph) + pynutil.insert("\"")

        final_graph = graph_dmy | graph_year
        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

    def _get_year_graph(self):
        """
        Transducer for year, e.g. twenty twenty -> 2020
        """

        def _get_thousands_graph():
            """
            ein tausend (neun hundert) [vierzehn/sechs und zwanzig/sieben]
            """
            graph_hundred_component = (graph_digit + delete_space + pynutil.delete("hundert")) | pynutil.insert("0")
            graph = (
                graph_digit
                + delete_space
                + pynutil.delete("tausend")
                + delete_space
                + graph_hundred_component
                + delete_space
                + (graph_teen | self.cardinal.graph_ties | (pynutil.insert("0") + graph_digit))
            )
            return graph

        def _get_hundreds_graph():
            """
            neunzehn hundert [vierzehn/sechs und zwanzig/sieben]
            """
            graph = (
                (graph_teen | self.cardinal.graph_ties)
                + delete_space
                + pynutil.delete("hundert")
                + delete_space
                + (graph_teen | self.cardinal.graph_ties | (pynutil.insert("0") + graph_digit))
            )
            return graph

        year_graph = (
            # 20 19, 40 12, 2012 - assuming no limit on the year
            ((graph_teen | self.cardinal.graph_ties) + delete_space + (self.cardinal.graph_ties | graph_teen))
            | _get_thousands_graph()
            | _get_hundreds_graph()
        )
        year_graph.optimize()
        return year_graph
