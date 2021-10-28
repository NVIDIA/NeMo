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

from nemo_text_processing.text_normalization.de.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    SINGULAR_TO_PLURAL,
    GraphFst,
    convert_space,
    delete_space,
    insert_space,
)

try:
    import pynini
    from pynini.lib import pynutil, rewrite

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, e.g.
        "5руб." -> money { "пять рублей" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph
        graph_decimal_final = decimal.final_graph_wo_negative

        unit_singular = pynini.string_file(get_abs_path("data/money/currency.tsv"))
        unit_plural = convert_space(unit_singular)  # @ SINGULAR_TO_PLURAL)
        unit_singular = convert_space(unit_singular)

        graph_unit_singular = pynutil.insert("currency: \"") + unit_singular + pynutil.insert("\"")
        graph_unit_plural = pynutil.insert("currency: \"") + unit_plural + pynutil.insert("\"")

        singular_graph = (
            graph_unit_singular
            + pynutil.insert(" integer_part: \"")
            + pynini.cross("1", pynini.union("eins", "ein", "einem", "eines", "einer", "einen"))
            + pynutil.insert("\"")
        )

        graph_decimal = graph_unit_plural + insert_space + graph_decimal_final

        graph_integer = (
            graph_unit_plural
            + pynutil.insert(" integer_part: \"")
            + ((NEMO_SIGMA - "1") @ cardinal_graph)
            + pynutil.insert("\"")
        )
        if not deterministic:
            graph_decimal |= singular_graph + insert_space + graph_decimal_final

        graph_integer |= singular_graph

        zero_graph = pynini.cross("0", "") | pynini.accep("0")
        # add minor currency part only when there are two digits after the point
        # .01 -> {zero one cent, one cent}, .05 -> {oh five, five cents}
        two_digits_fractional_part = (
            NEMO_SIGMA
            + pynini.closure(NEMO_DIGIT)
            + (
                (pynini.accep(",") + (NEMO_DIGIT ** (2) | zero_graph + (NEMO_DIGIT - "0")))
                | pynutil.delete(",") + pynini.cross(pynini.closure("0", 1), "")
            )
        )
        currencies = load_labels(get_abs_path("data/money/currency.tsv"))
        decimal_graph_with_minor = None
        for curr_symbol, curr_name in currencies:
            curr_symbol_graph = pynutil.delete(curr_symbol)
            graph_end_maj = pynutil.insert(" currency_maj: \"" + curr_symbol + "\"")
            graph_end_min = pynutil.insert(" currency_min: \"" + curr_symbol + "\"")
            preserve_order = pynutil.insert(" preserve_order: true")
            integer_part = decimal.graph_integer + graph_end_maj

            minor_curr = (
                (NEMO_DIGIT + NEMO_DIGIT) | (pynutil.delete("0") + NEMO_DIGIT)
            ) @ cardinal.graph_hundred_component_at_least_one_none_zero_digit
            minor_curr = pynutil.insert("fractional_part: \"") + minor_curr + pynutil.insert("\"")

            decimal_graph_with_minor_curr = (
                curr_symbol_graph
                + pynini.closure(integer_part, 0, 1)
                + pynini.cross(",", " ")
                + minor_curr
                + graph_end_min
                + preserve_order
            )

            decimal_graph_with_minor_curr = pynini.compose(two_digits_fractional_part, decimal_graph_with_minor_curr)

            decimal_graph_with_minor = (
                decimal_graph_with_minor_curr
                if decimal_graph_with_minor is None
                else pynini.union(decimal_graph_with_minor, decimal_graph_with_minor_curr)
            )

        graph_decimal |= pynutil.add_weight(decimal_graph_with_minor, -0.01)
        final_graph = graph_integer | graph_decimal

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
