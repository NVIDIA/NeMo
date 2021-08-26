# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.de.graph_utils import GraphFst, delete_extra_space, delete_space
from nemo_text_processing.inverse_text_normalization.de.taggers.cardinal import AND
from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path

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

    PYNINI_AVAILABLE = False


def _get_month_graph():
    """
    Transducer for month, e.g. april -> april
    """
    month_graph = pynini.string_file(get_abs_path("data/months.tsv"))
    return month_graph


def _get_digit_or_teen():
    """
    Transducer for single digit or teens
    """
    return (
        pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        | pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
    ).optimize()


def _get_single_digit():
    """
    Transducer for single digit
    """
    return pynini.string_file(get_abs_path("data/numbers/digit.tsv")).optimize()


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, in the form of (day) month (year) or year
        e.g. vierundzwanzigster juli zwei tausend dreizehn -> date { day: "24" month: "juli" year: "2013" preserve_order: true }
        e.g. neunzehnachtzig -> date { year: "1980" preserve_order: true }
        e.g. neunzehnachtziger -> date { year: "1980er" preserve_order: true }
        e.g. neunzehnhundertundachtzig -> date { year: "1980" preserve_order: true }
        e.g. vierzehnter januar -> date { day: "24" month: "januar"  preserve_order: true }
        e.g. zwanzig zwanzig -> date { year: "2020" preserve_order: true }

    Args:
        ordinal: OrdinalFst
        cardinal: CardinalFst
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
        optional_graph_year = pynini.closure(
            delete_extra_space
            + pynutil.insert("year: \"")
            + pynutil.add_weight(year_graph, -YEAR_WEIGHT)
            + pynutil.insert("\""),
            0,
            1,
        )
        graph_dmy = day_graph + delete_extra_space + month_graph + optional_graph_year
        graph_year = (
            pynutil.insert("year: \"")
            + year_graph
            + pynini.closure(pynini.accep('er') + pynini.closure(pynini.accep('n'), 0, 1), 0, 1)
            + pynutil.insert("\"")
        )

        final_graph = graph_dmy | graph_year
        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

    def _get_year_graph(self):
        """
        Transducer for year
        """

        def _get_graph():
            """
            ein tausend (elf hundert) [vierzehn/sechs und zwanzig/sieben]
            """
            graph_hundred_prefix = (
                _get_digit_or_teen()
                + delete_space
                + pynutil.delete("hundert")
                + pynini.closure(delete_space + pynutil.delete(AND), 0, 1)
            )
            graph_thousand_prefix = (
                _get_single_digit()
                + delete_space
                + pynutil.delete("tausend")
                + pynini.closure(delete_space + pynutil.delete(AND), 0, 1)
                + pynutil.insert('0')
            )
            graph = (
                pynini.union(graph_hundred_prefix, graph_thousand_prefix)
                + delete_space
                + (graph_teen | self.cardinal.graph_ties | (pynutil.insert("0") + graph_digit))
            )
            return graph

        year_graph = (
            # 20 19, 40 12, 2012 - assuming no limit on the year
            ((graph_teen | self.cardinal.graph_ties) + delete_space + (self.cardinal.graph_ties | graph_teen))
            | _get_graph()
        )
        year_graph.optimize()
        return year_graph
