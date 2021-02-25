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

import csv
import os

import pynini
from denormalization.data_loader_utils import get_abs_path
from denormalization.graph_utils import NEMO_SPACE, GraphFst, add_arcs, delete_space
from denormalization.taggers.ordinal import OrdinalFst
from pynini.lib import pynutil

_measurements_tsv = open(os.path.dirname(os.path.abspath(__file__)) + "/../data/months.tsv")
read_tsv = csv.reader(_measurements_tsv, delimiter="\t")
_measurements_dict = dict(read_tsv)
_measurements_dict = {v: k for k, v in _measurements_dict.items()}


def _get_month_graph():
    months_labels = []
    for mon in _measurements_dict.keys():
        months_labels.append([mon])
        months_labels.append([mon.lower()])

    month_graph = pynini.Fst()
    s_start = month_graph.add_state()
    s_final = month_graph.add_state()
    month_graph.set_start(s_start)
    month_graph.set_final(s_final)

    add_arcs(month_graph, s_start, s_final, months_labels)
    month_graph.optimize()
    return month_graph


def _get_year_graph():
    graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).optimize()
    ties_graph = pynini.string_file(get_abs_path("data/numbers/ties.tsv")).optimize()
    graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv")).optimize()
    # graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))

    def _get_digits_graph():
        zero = pynini.cross((pynini.accep("oh") | pynini.accep("o")), "0")
        graph = zero + delete_space + graph_digit
        graph.optimize()

        return graph

    def _get_ties_graph():
        graph = ties_graph + (delete_space + graph_digit | pynini.cross("", "0"))
        return graph

    def _get_thousands_graph():
        graph_ties = _get_ties_graph()
        # graph_hundred_component = pynini.union(graph_digit + space + graph_hundred, pynutil.insert("0"))
        graph_hundred_component = (graph_digit + delete_space + pynini.cross("hundred", "")) | pynutil.insert("0")

        # graph_thousands = pynini.union(
        #     graph_hundred_component_at_least_one_none_zero_digit + space + pynini.cross("thousand", ""),
        #     pynutil.insert("000"))
        graph = (
            graph_digit
            + delete_space
            + pynini.cross("thousand", "")
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
        # 20 19, 40 12 - assuming no limit on the year
        (graph_teen + delete_space + (graph_ties | graph_digits | graph_teen))
        | (graph_ties + delete_space + (graph_ties | graph_digits | graph_teen))
        | graph_thousands
    )
    year_graph.optimize()
    return year_graph


class DateFst(GraphFst):
    def __init__(self):
        super().__init__(name="date", kind="classify")
        # weekday, day, month, year, style(depr), text(depr), short_year(depr), era

        year_graph = _get_year_graph()
        month_graph = _get_month_graph()

        month_graph = pynini.closure(
            pynutil.insert("month: \"") + month_graph + pynutil.insert("\"") + NEMO_SPACE, 0, 1
        )
        day_graph = OrdinalFst().graph
        day_graph = pynini.closure(pynutil.insert("day: \"") + day_graph + pynutil.insert("\"") + NEMO_SPACE, 0, 1)
        graph1 = month_graph + day_graph + pynutil.insert("year: \"") + year_graph + pynutil.insert("\"")
        graph2 = (
            pynini.cross("the", "")
            + delete_space
            + day_graph
            + pynini.cross("of", "")
            + delete_space
            + month_graph
            + pynutil.insert("year: \"")
            + year_graph
            + pynutil.insert("\"")
        )

        final_graph = graph1 | graph2
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
