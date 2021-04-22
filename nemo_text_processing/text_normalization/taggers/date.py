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
from nemo_text_processing.text_normalization.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)

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


def _get_ties_graph():
    graph = (
        graph_teen
        | ties_graph + pynutil.delete("0")
        | ties_graph + pynutil.insert(" ") + graph_digit
        | pynini.cross("0", "o") + pynutil.insert(" ") + graph_digit
    )
    return graph.optimize()


def _get_year_graph():
    """
    1290-> twelve nineteen, only from 1000 - 2999
    """

    graph_ties = _get_ties_graph()
    graph = (
        graph_ties + pynutil.insert(" ") + graph_ties
        | graph_ties + pynutil.insert(" ") + pynini.cross("00", "hundred")
        | pynini.cross("2", "two") + pynutil.insert(" ") + pynini.cross("000", "thousand")
        | (graph_teen + pynutil.insert(" ") + ties_graph + pynutil.delete("0s"))
        @ pynini.cdrewrite(pynini.cross("y", "ies"), "", "[EOS]", NEMO_SIGMA)
    )
    graph = (pynini.union("1", "2") + NEMO_DIGIT + NEMO_DIGIT + NEMO_DIGIT + pynini.closure("s", 0, 1)) @ graph
    return graph


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, 
        e.g. january fifth twenty twelve -> date { month: "january" day: "5" year: "2012" preserve_order: true }
        e.g. the fifth of january twenty twelve -> date { day: "5" month: "january" year: "2012" preserve_order: true }

    Args:
        ordinal: Ordinal GraphFST
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="date", kind="classify")

        month_graph = pynini.string_file(get_abs_path("data/months.tsv")).optimize()
        month_numbers_graph = pynini.string_file(get_abs_path("data/months_numbers.tsv")).optimize()

        cardinal_graph = cardinal.graph_hundred_component_at_least_one_none_zero_digit

        # weekday, day, month, year, style(depr), text(depr), short_year(depr), era
        year_graph = _get_year_graph()

        YEAR_WEIGHT = 0.001
        year_graph_standalone = (
            pynutil.insert("year: \"") + pynutil.add_weight(year_graph, YEAR_WEIGHT) + pynutil.insert("\"")
        )

        month_graph = pynutil.insert("month: \"") + month_graph + pynutil.insert("\"")
        month_numbers_graph = pynutil.insert("month: \"") + month_numbers_graph + pynutil.insert("\"")

        day_graph = pynutil.insert("day: \"") + cardinal_graph + pynutil.insert("\"")
        optional_day_graph = pynini.closure(delete_extra_space + day_graph, 0, 1)

        year_graph = pynutil.insert("year: \"") + year_graph + pynutil.insert("\"")
        optional_graph_year = pynini.closure(delete_extra_space + year_graph, 0, 1,)
        graph_mdy = (
            month_graph
            + optional_day_graph
            + delete_space
            + pynini.closure(pynutil.delete(","), 0, 1)
            + optional_graph_year
        )
        graph_dmy = day_graph + delete_extra_space + month_graph + optional_graph_year
        delete_sep = pynutil.delete(pynini.union("-", "/"))
        graph_ymd = (
            year_graph
            + delete_sep
            + insert_space
            + month_numbers_graph
            + delete_sep
            + insert_space
            + pynini.closure(pynutil.delete("0"), 0, 1)
            + day_graph
        )

        final_graph = (graph_mdy | graph_dmy) + pynutil.insert(" preserve_order: true")
        final_graph |= graph_ymd | year_graph_standalone
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
