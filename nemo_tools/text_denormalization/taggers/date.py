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

import pynini
from nemo_tools.text_denormalization.data_loader_utils import get_abs_path
from nemo_tools.text_denormalization.graph_utils import NEMO_SIGMA, GraphFst, delete_extra_space, delete_space
from nemo_tools.text_denormalization.taggers.ordinal import OrdinalFst
from pynini.lib import pynutil


def _get_month_graph():
    month_graph = pynini.string_file(get_abs_path("data/months.tsv"))
    month_graph = pynini.invert(month_graph).optimize()
    return month_graph


graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv")).optimize()
graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).optimize()
ties_graph = pynini.string_file(get_abs_path("data/numbers/ties.tsv")).optimize()


def _get_ties_graph():
    graph = ties_graph + (delete_space + graph_digit | pynutil.insert("0"))
    return graph


def _get_range_graph():
    graph_ties = _get_ties_graph()
    graph = (graph_ties | graph_teen) + delete_space + pynini.cross("hundreds", "00s")
    graph |= pynini.cross("two", "2") + delete_space + pynini.cross("thousands", "000s")
    graph |= (
        (graph_ties | graph_teen)
        + delete_space
        + (NEMO_SIGMA + pynini.cross("ties", "ty")) @ graph_ties
        + pynutil.insert("s")
    )
    return graph


def _get_year_graph():
    def _get_digits_graph():
        zero = pynini.cross((pynini.accep("oh") | pynini.accep("o")), "0")
        graph = zero + delete_space + graph_digit
        graph.optimize()
        return graph

    def _get_thousands_graph():
        graph_ties = _get_ties_graph()
        graph_hundred_component = (graph_digit + delete_space + pynutil.delete("hundred")) | pynutil.insert("0")
        graph = (
            graph_digit
            + delete_space
            + pynutil.delete("thousand")
            + delete_space
            + graph_hundred_component
            + delete_space
            + (graph_teen | graph_ties)
        )
        return graph

    graph_ties = _get_ties_graph()
    graph_digits = _get_digits_graph()
    graph_thousands = _get_thousands_graph()
    year_graph = (
        # 20 19, 40 12, 2012 - assuming no limit on the year
        (graph_teen + delete_space + (graph_ties | graph_digits | graph_teen))
        | (graph_ties + delete_space + (graph_ties | graph_digits | graph_teen))
        | graph_thousands
    )
    year_graph.optimize()
    return year_graph


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, 
        e.g. january fifth twenty twelve -> date { month: "january" day: "5" year: "2012" preserve_order: true }
        e.g. the fifth of january twenty twelve -> date { day: "5" month: "january" year: "2012" preserve_order: true }
    """

    def __init__(self):
        super().__init__(name="date", kind="classify")
        # weekday, day, month, year, style(depr), text(depr), short_year(depr), era
        year_graph = _get_year_graph()
        YEAR_WEIGHT = 0.001
        year_graph = pynutil.add_weight(year_graph, YEAR_WEIGHT)
        month_graph = _get_month_graph()

        month_graph = pynutil.insert("month: \"") + month_graph + pynutil.insert("\"")
        day_graph = pynutil.insert("day: \"") + pynutil.add_weight(OrdinalFst().graph, -0.7) + pynutil.insert("\"")
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
        graph_dmy = (
            pynutil.delete("the")
            + delete_space
            + day_graph
            + delete_space
            + pynutil.delete("of")
            + delete_extra_space
            + month_graph
            + optional_graph_year
        )
        graph_year = pynutil.insert("year: \"") + (year_graph | _get_range_graph()) + pynutil.insert("\"")

        final_graph = graph_mdy | graph_dmy | graph_year
        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
