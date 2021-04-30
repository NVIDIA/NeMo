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
from nemo_text_processing.text_normalization.graph_utils import NEMO_ALPHA, NEMO_DIGIT, GraphFst, delete_extra_space

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g. 
        -23 -> cardinal { negative: "true"  integer: "twenty three" } }
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")

        delete_space = pynutil.delete(" ")
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))

        delete_extra_spaces = (
            pynini.closure(pynutil.delete(" "))
            + pynini.closure(pynini.closure(NEMO_ALPHA, 1) + delete_extra_space)
            + pynini.closure(NEMO_ALPHA, 1)
            + pynini.closure(pynutil.delete(" "))
        )

        graph_hundred = pynutil.delete("hundred")

        graph_hundred_component = pynini.union(
            graph_digit + delete_space + graph_hundred + delete_space, pynutil.insert("0")
        )
        graph_hundred_component += pynini.union(
            graph_teen | pynutil.insert("00"),
            (graph_ties + delete_space | pynutil.insert("0")) + (graph_digit | pynutil.insert("0")),
        )

        #  string -> all 3 digit numbers apart from 000
        graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
            pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)
        )

        # all 3 digit numbers apart from 0 -> string
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            pynini.invert(
                graph_hundred_component_at_least_one_none_zero_digit
                @ (
                    pynutil.delete(pynini.closure("0"))
                    + pynini.difference(NEMO_DIGIT, "0")
                    + pynini.closure(NEMO_DIGIT)
                )
            )
            @ delete_extra_spaces
        ).optimize()

        insert_comma = pynini.closure(pynutil.insert(","), 0, 1)

        graph_thousands = (
            pynini.union(
                graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("thousand"),
                pynutil.insert("000", weight=0.1),
            )
            + insert_comma
        )

        graph_million = (
            pynini.union(
                graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("million"),
                pynutil.insert("000", weight=0.1),
            )
            + insert_comma
        )
        graph_billion = (
            pynini.union(
                graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("billion"),
                pynutil.insert("000", weight=0.1),
            )
            + insert_comma
        )
        graph_trillion = (
            pynini.union(
                graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("trillion"),
                pynutil.insert("000", weight=0.1),
            )
            + insert_comma
        )
        graph_quadrillion = (
            pynini.union(
                graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("quadrillion"),
                pynutil.insert("000", weight=0.1),
            )
            + insert_comma
        )
        graph_quintillion = (
            pynini.union(
                graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("quintillion"),
                pynutil.insert("000", weight=0.1),
            )
            + insert_comma
        )
        graph_sextillion = (
            pynini.union(
                graph_hundred_component_at_least_one_none_zero_digit + delete_space + pynutil.delete("sextillion"),
                pynutil.insert("000", weight=0.1),
            )
            + insert_comma
        )

        graph = pynini.union(
            graph_sextillion
            + delete_space
            + graph_quintillion
            + delete_space
            + graph_quadrillion
            + delete_space
            + graph_trillion
            + delete_space
            + graph_billion
            + delete_space
            + graph_million
            + delete_space
            + graph_thousands
            + delete_space
            + graph_hundred_component,
            graph_zero,
        )

        graph = graph @ pynini.union(
            pynini.closure(pynutil.delete(pynini.union("0", ",")))
            + pynini.difference(NEMO_DIGIT, "0")
            + pynini.closure(pynini.union(NEMO_DIGIT, ",")),
            "0",
        )

        self.graph = pynini.invert(graph) @ delete_extra_spaces
        self.graph = self.graph.optimize()

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
