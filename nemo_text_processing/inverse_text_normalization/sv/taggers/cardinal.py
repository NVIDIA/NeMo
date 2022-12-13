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


import pynini
from nemo_text_processing.inverse_text_normalization.sv.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. minus tjugotre -> cardinal { negative: "-" integer: "23"} 
    This class converts cardinals up to (but not including) "quadriljon",
    i.e up to "one septillion" in English (10^{24}).
    Cardinals below ten are not converted (in order to avoid 
    "ett hus" --> "1 hus" and any other odd conversions.)
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        graph_hundreds = pynini.string_file(get_abs_path("data/numbers/hundreds.tsv"))

        graph_hundred_component = graph_hundreds | pynutil.insert("0")
        graph_hundred_component += pynini.union(
            graph_teen | pynutil.insert("00"),
            (graph_ties | pynutil.insert("0")) + (graph_digit | pynutil.insert("0")),
        )

        graph_hundred_component_at_least_one_none_zero_digit = graph_hundred_component @ (
            pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)
        )
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )

        graph_thousands = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit + pynutil.delete("tusen"),
            pynutil.insert("000", weight=0.1),
        )

        graph_miljoner = (
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + (pynutil.delete("miljoner") | pynutil.delete("miljon"))
        )

        graph_miljarder = (
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + (pynutil.delete("miljarder") | pynutil.delete("miljard"))
        )

        graph_biljoner = (
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + (pynutil.delete("biljoner") | pynutil.delete("biljon"))
        )

        graph_biljarder = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + (pynutil.delete("biljarder") | pynutil.delete("biljard"))
        )

        graph_triljoner = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + (pynutil.delete("triljoner") | pynutil.delete("triljon")),
        )

        graph_triljarder = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + (pynutil.delete("triljarder") | pynutil.delete("triljard")),
        )

        graph = pynini.union(
            (graph_triljarder | pynutil.insert("000", weight=0.1) + graph_triljoner)
            + delete_space
            + (graph_biljarder | pynutil.insert("000", weight=0.1) + graph_biljoner)
            + delete_space
            + pynini.union(
                graph_miljarder,
                pynutil.insert("000", weight=0.1) + graph_miljoner,
                graph_miljarder + graph_miljoner,
                graph_miljarder + pynutil.insert("000", weight=0.1),
            )
            + delete_space
            + graph_thousands
            + delete_space
            + graph_hundred_component,
            graph_zero,
        )

        graph = graph @ pynini.union(
            pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT), "0"
        )

        self.graph_no_exception = graph

        # save self.numbers_up_to_thousand for use in DecimalFst
        digits_up_to_thousand = NEMO_DIGIT | (NEMO_DIGIT ** 2) | (NEMO_DIGIT ** 3)
        numbers_up_to_thousand = pynini.compose(graph, digits_up_to_thousand).optimize()
        self.numbers_up_to_thousand = numbers_up_to_thousand

        # save self.numbers_up_to_million for use in DecimalFst
        digits_up_to_million = (
            NEMO_DIGIT
            | (NEMO_DIGIT ** 2)
            | (NEMO_DIGIT ** 3)
            | (NEMO_DIGIT ** 4)
            | (NEMO_DIGIT ** 5)
            | (NEMO_DIGIT ** 6)
        )
        numbers_up_to_million = pynini.compose(graph, digits_up_to_million).optimize()
        self.numbers_up_to_million = numbers_up_to_million

        # don't convert cardinals from zero to nine inclusive
        graph_exception = pynini.project(pynini.union(graph_digit, graph_zero), 'input')

        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus", "\"-\"") + NEMO_SPACE, 0, 1
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
