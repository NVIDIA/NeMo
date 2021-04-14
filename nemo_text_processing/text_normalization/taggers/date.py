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

from nemo_text_processing.text_normalization.data_loader_utils import get_abs_path
from nemo_text_processing.text_normalization.graph_utils import NEMO_SIGMA, GraphFst, delete_extra_space, delete_space

try:
    import pynini
    from pynini.lib import pynutil

    graph_teen = pynini.invert(pynini.string_file(get_abs_path("data/numbers/teen.tsv"))).optimize()
    graph_digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv"))).optimize()
    ties_graph = pynini.invert(pynini.string_file(get_abs_path("data/numbers/ties.tsv"))).optimize()

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    # Add placeholders for global variables
    graph_teen = None
    graph_digit = None
    ties_graph = None

    PYNINI_AVAILABLE = True


def _get_month_graph():
    month_graph = pynini.string_file(get_abs_path("data/months.tsv")).optimize()
    return month_graph


def _get_ties_graph():
    graph = (
        graph_teen
        | ties_graph + pynutil.delete("0")
        | ties_graph + pynutil.insert(" ") + graph_digit
        | pynini.cross("0", "o") + pynutil.insert(" ") + graph_digit
    )
    return graph.optimize()


def _get_year_graph():

    graph_ties = _get_ties_graph()
    graph = (
        graph_ties + pynutil.insert(" ") + graph_ties
        | graph_ties + pynutil.insert(" ") + pynini.cross("00", "hundred")
        | pynini.cross("2", "two") + pynutil.insert(" ") + pynini.cross("000", "thousand")
    )
    return graph


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, 
        e.g. january fifth twenty twelve -> date { month: "january" day: "5" year: "2012" preserve_order: true }
        e.g. the fifth of january twenty twelve -> date { day: "5" month: "january" year: "2012" preserve_order: true }

    Args:
        ordinal: Ordinal GraphFST
    """

    def __init__(self, ordinal: GraphFst):
        super().__init__(name="date", kind="classify")

        ordinal_graph = ordinal.graph

        # weekday, day, month, year, style(depr), text(depr), short_year(depr), era
        year_graph = _get_year_graph()
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
        graph_mdy = month_graph + optional_day_graph + optional_graph_year
        graph_dmy = day_graph + delete_extra_space + month_graph + optional_graph_year
        graph_year = pynutil.insert("year: \"") + year_graph + pynutil.insert("\"")

        final_graph = graph_mdy | graph_dmy | graph_year
        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
