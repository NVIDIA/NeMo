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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    SINGULAR_TO_PLURAL,
    GraphFst,
    convert_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.en.taggers.date import get_hundreds_graph
from nemo_text_processing.text_normalization.en.utils import get_abs_path, load_labels

try:
    import pynini
    from pynini.lib import pynutil, rewrite

    min_singular = pynini.string_file(get_abs_path("data/currency/currency_minor_singular.tsv"))
    min_plural = pynini.string_file(get_abs_path("data/currency/currency_minor_plural.tsv"))
    maj_singular = pynini.string_file((get_abs_path("data/currency/currency.tsv")))

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, suppletive aware, e.g. 
        $12.05 -> money { integer_part: "twelve" currency_maj: "dollars" fractional_part: "five" currency_min: "cents" preserve_order: true }
        $12.0500 -> money { integer_part: "twelve" currency_maj: "dollars" fractional_part: "five" currency_min: "cents" preserve_order: true }
        $1 -> money { currency_maj: "dollar" integer_part: "one" }
        $1.00 -> money { currency_maj: "dollar" integer_part: "one" }
        $0.05 -> money { fractional_part: "five"  currency_min: "cents" preserve_order: true }
        $1 million -> money { currency_maj: "dollars" integer_part: "one" quantity: "million" }
        $1.2 million -> money { currency_maj: "dollars" integer_part: "one"  fractional_part: "two" quantity: "million" }
        $1.2320 -> money { currency_maj: "dollars" integer_part: "one"  fractional_part: "two three two" }

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

        maj_singular_labels = load_labels(get_abs_path("data/currency/currency.tsv"))
        maj_unit_plural = convert_space(maj_singular @ SINGULAR_TO_PLURAL)
        maj_unit_singular = convert_space(maj_singular)

        graph_maj_singular = pynutil.insert("currency_maj: \"") + maj_unit_singular + pynutil.insert("\"")
        graph_maj_plural = pynutil.insert("currency_maj: \"") + maj_unit_plural + pynutil.insert("\"")

        optional_delete_fractional_zeros = pynini.closure(
            pynutil.delete(".") + pynini.closure(pynutil.delete("0"), 1), 0, 1
        )

        graph_integer_one = pynutil.insert("integer_part: \"") + pynini.cross("1", "one") + pynutil.insert("\"")
        # only for decimals where third decimal after comma is non-zero or with quantity
        decimal_delete_last_zeros = (
            pynini.closure(NEMO_DIGIT)
            + pynini.accep(".")
            + pynini.closure(NEMO_DIGIT, 2)
            + (NEMO_DIGIT - "0")
            + pynini.closure(pynutil.delete("0"))
        )
        decimal_with_quantity = NEMO_SIGMA + NEMO_ALPHA

        graph_decimal = (
            graph_maj_plural + insert_space + (decimal_delete_last_zeros | decimal_with_quantity) @ graph_decimal_final
        )

        if deterministic:
            graph_integer = (
                pynutil.insert("integer_part: \"") + ((NEMO_SIGMA - "1") @ cardinal_graph) + pynutil.insert("\"")
            )
        else:
            graph_integer = (
                pynutil.insert("integer_part: \"")
                + ((NEMO_SIGMA - "1") @ (get_hundreds_graph(deterministic) | cardinal_graph))
                + pynutil.insert("\"")
            )
            graph_decimal |= graph_maj_singular + insert_space + graph_integer_one + insert_space + graph_decimal_final

        graph_integer_only = graph_maj_singular + insert_space + graph_integer_one
        graph_integer_only |= graph_maj_plural + insert_space + graph_integer

        final_graph = (graph_integer_only + optional_delete_fractional_zeros) | graph_decimal

        # remove trailing zeros of non zero number in the first 2 digits and fill up to 2 digits
        # e.g. 2000 -> 20, 0200->02, 01 -> 01, 10 -> 10
        # not accepted: 002, 00, 0,
        two_digits_fractional_part = (
            pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(pynutil.delete("0"))
        ) @ (
            (pynutil.delete("0") + (NEMO_DIGIT - "0"))
            | ((NEMO_DIGIT - "0") + pynutil.insert("0"))
            | ((NEMO_DIGIT - "0") + NEMO_DIGIT)
        )

        graph_min_singular = pynutil.insert(" currency_min: \"") + min_singular + pynutil.insert("\"")
        graph_min_plural = pynutil.insert(" currency_min: \"") + min_plural + pynutil.insert("\"")
        # format ** dollars ** cent
        decimal_graph_with_minor = None
        for curr_symbol, _ in maj_singular_labels:
            preserve_order = pynutil.insert(" preserve_order: true")
            integer_plus_maj = graph_integer + insert_space + pynutil.insert(curr_symbol) @ graph_maj_plural
            integer_plus_maj |= graph_integer_one + insert_space + pynutil.insert(curr_symbol) @ graph_maj_singular
            # non zero integer part
            integer_plus_maj = (pynini.closure(NEMO_DIGIT) - "0") @ integer_plus_maj

            graph_fractional_one = two_digits_fractional_part @ pynini.cross("1", "one")
            graph_fractional_one = pynutil.insert("fractional_part: \"") + graph_fractional_one + pynutil.insert("\"")
            graph_fractional = (
                two_digits_fractional_part
                @ (pynini.closure(NEMO_DIGIT, 1, 2) - "1")
                @ cardinal.graph_hundred_component_at_least_one_none_zero_digit
            )
            graph_fractional = pynutil.insert("fractional_part: \"") + graph_fractional + pynutil.insert("\"")

            fractional_plus_min = graph_fractional + insert_space + pynutil.insert(curr_symbol) @ graph_min_plural
            fractional_plus_min |= (
                graph_fractional_one + insert_space + pynutil.insert(curr_symbol) @ graph_min_singular
            )

            decimal_graph_with_minor_curr = integer_plus_maj + pynini.cross(".", " ") + fractional_plus_min

            if not deterministic:
                decimal_graph_with_minor_curr |= pynutil.add_weight(
                    integer_plus_maj
                    + pynini.cross(".", " ")
                    + pynutil.insert("fractional_part: \"")
                    + two_digits_fractional_part @ cardinal.graph_hundred_component_at_least_one_none_zero_digit
                    + pynutil.insert("\""),
                    weight=0.0001,
                )
            decimal_graph_with_minor_curr |= (
                pynini.closure(pynutil.delete("0"), 0, 1) + pynutil.delete(".") + fractional_plus_min
            )
            decimal_graph_with_minor_curr = (
                pynutil.delete(curr_symbol) + decimal_graph_with_minor_curr + preserve_order
            )

            decimal_graph_with_minor = (
                decimal_graph_with_minor_curr
                if decimal_graph_with_minor is None
                else pynini.union(decimal_graph_with_minor, decimal_graph_with_minor_curr)
            )

        final_graph |= decimal_graph_with_minor

        # to be refactored
        if not deterministic:
            unit_singular = pynini.string_file(get_abs_path("data/currency/currency.tsv"))
            unit_plural = convert_space(unit_singular @ SINGULAR_TO_PLURAL)
            unit_singular = convert_space(unit_singular)

            graph_unit_singular = pynutil.insert("currency: \"") + unit_singular + pynutil.insert("\"")
            graph_unit_plural = pynutil.insert("currency: \"") + unit_plural + pynutil.insert("\"")

            singular_graph = (
                graph_unit_singular
                + pynutil.insert(" integer_part: \"")
                + pynini.cross("1", "one")
                + pynutil.insert("\"")
            )

            graph_decimal = graph_unit_plural + insert_space + graph_decimal_final
            graph_integer = (
                graph_unit_plural
                + pynutil.insert(" integer_part: \"")
                + ((NEMO_SIGMA - "1") @ (get_hundreds_graph(deterministic) | cardinal_graph))
                + pynutil.insert("\"")
            )
            graph_decimal |= singular_graph + insert_space + graph_decimal_final

            graph_integer |= singular_graph

            final_graph = graph_integer | graph_decimal

            currencies = maj_singular_labels
            zero_graph = pynini.cross("0", "") | pynini.accep("0")
            # add minor currency part only when there are two digits after the point
            # .01 -> {zero one cent, one cent}, .05 -> {oh five, five cents}
            two_digits_fractional_part = (
                NEMO_SIGMA
                + pynini.closure(NEMO_DIGIT)
                + (
                    (pynini.accep(".") + (NEMO_DIGIT ** (2) | zero_graph + (NEMO_DIGIT - "0")))
                    | pynutil.delete(".") + pynini.cross(pynini.closure("0", 1), "")
                )
            )

            integer_graph = None
            decimal_graph_with_minor = None
            decimal_graph_default = None

            for curr_symbol, curr_name in currencies:
                curr_symbol_graph = pynutil.delete(curr_symbol)
                graph_end = pynutil.insert(" currency: \"" + curr_symbol + "\"")
                preserve_order = pynutil.insert(" preserve_order: True")
                integer_part = decimal.graph_integer + graph_end + preserve_order

                # "$4" -> 'integer_part: "four" currency: "$" preserve_order: True' -> four dollars
                integer_graph_curr = curr_symbol_graph + integer_part
                # remove fractional part if it contains only zeros
                # "$4.00" -> 'integer_part: "four" currency: "$" preserve_order: True' -> four dollars
                integer_graph_curr |= pynini.compose(two_digits_fractional_part, integer_graph_curr)
                decimal_graph_with_minor_curr = (
                    curr_symbol_graph
                    + pynini.closure(integer_part, 0, 1)
                    + pynini.cross(".", " ")
                    + decimal.graph_fractional
                    + graph_end
                )

                # "$.5" -> 'fractional_part: "five" currency: "dollars"' -> point five dollars
                decimal_graph_default_curr = (
                    pynutil.delete("currency: \"" + pynini.compose(curr_symbol, unit_plural) + "\"")
                    + delete_space
                    + pynini.accep("fractional_part")
                    + NEMO_SIGMA
                    + pynutil.insert(" currency: \"" + pynini.compose(curr_symbol, unit_plural) + "\"")
                )

                # "$4.5" -> 'integer_part: "four" fractional_part: "five" currency: "dollars"' -> "four point five dollars"
                decimal_graph_default_curr |= (
                    pynutil.delete("currency: \"" + curr_name + pynini.closure(NEMO_NOT_QUOTE) + "\"")
                    + delete_space
                    + pynini.accep("integer_part")
                    + NEMO_SIGMA
                    + pynini.accep("fractional_part")
                    + NEMO_SIGMA
                    + pynutil.insert(" currency: \"" + pynini.compose(curr_symbol, unit_plural) + "\"")
                )

                # "£4 billion" -> 'integer_part: "four" quantity: "billion" currency: "pounds"' -> "four billion dollars"
                decimal_graph_default_curr |= (
                    pynutil.delete("currency: \"")
                    + pynutil.delete(
                        rewrite.rewrite_lattice(curr_symbol, pynini.compose(curr_symbol, unit_plural)) + "\" "
                    )
                    + pynini.difference(NEMO_SIGMA, "fractional_part")
                    + pynutil.insert(" currency: \"" + pynini.compose(curr_symbol, unit_plural) + "\"")
                )

                decimal_graph_with_minor_curr = pynini.compose(
                    two_digits_fractional_part, decimal_graph_with_minor_curr
                )
                decimal_graph_default_curr = pynini.compose(graph_decimal, decimal_graph_default_curr)

                integer_graph = (
                    integer_graph_curr if integer_graph is None else pynini.union(integer_graph, integer_graph_curr)
                )
                decimal_graph_with_minor = (
                    decimal_graph_with_minor_curr
                    if decimal_graph_with_minor is None
                    else pynini.union(decimal_graph_with_minor, decimal_graph_with_minor_curr)
                )
                decimal_graph_default = (
                    decimal_graph_default_curr
                    if decimal_graph_default is None
                    else pynini.union(decimal_graph_default, decimal_graph_default_curr)
                )

            final_graph = decimal_graph_with_minor | decimal_graph_default | integer_graph

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
