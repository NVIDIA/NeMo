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
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    PLURAL_TO_SINGULAR,
    SINGULAR_TO_PLURAL,
    GraphFst,
    convert_space,
    insert_space,
)
from nemo_text_processing.text_normalization.en.taggers.date import get_hundreds_graph
from nemo_text_processing.text_normalization.en.utils import get_abs_path, load_labels

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, suppletive aware, e.g. 
        $12.05 -> money { currency: "dollars" integer_part: "twelve" fractional_part: "o five" }
        $1 -> money { currency: "dollar" integer_part: "one" }

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

        unit_singular = pynini.string_file(get_abs_path("data/currency/currency.tsv"))
        unit_plural = convert_space(unit_singular @ SINGULAR_TO_PLURAL)
        unit_singular = convert_space(unit_singular)

        graph_unit_singular = pynutil.insert("currency: \"") + unit_singular + pynutil.insert("\"")
        graph_unit_plural = pynutil.insert("currency: \"") + unit_plural + pynutil.insert("\"")

        singular_graph = (
            graph_unit_singular + pynutil.insert(" integer_part: \"") + pynini.cross("1", "one") + pynutil.insert("\"")
        )

        graph_decimal = graph_unit_plural + insert_space + graph_decimal_final

        if deterministic:
            graph_integer = (
                graph_unit_plural
                + pynutil.insert(" integer_part: \"")
                + ((NEMO_SIGMA - "1") @ cardinal_graph)
                + pynutil.insert("\"")
            )
        else:
            graph_integer = (
                graph_unit_plural
                + pynutil.insert(" integer_part: \"")
                + ((NEMO_SIGMA - "1") @ (get_hundreds_graph(deterministic) | cardinal_graph))
                + pynutil.insert("\"")
            )
            graph_decimal |= singular_graph + insert_space + graph_decimal_final

        graph_integer |= singular_graph

        final_graph = graph_integer | graph_decimal

        # if not deterministic:
        #     # currency = pynini.project(graph_unit_plural, "input")
        #     minor_singular = pynini.string_file(get_abs_path("data/currency/currency_minor_singular.tsv"))
        #     minor_plural = pynini.string_file(get_abs_path("data/currency/currency_minor_plural.tsv"))
        #
        #     # currency_part at the beginning removed
        #     graph_decimal_no_currency = pynini.compose(
        #         graph_decimal, pynutil.delete("currency: \"" + pynini.closure(NEMO_NOT_QUOTE, 1) + "\" ") + NEMO_SIGMA
        #     )
        #
        #     # move the major currency after decimal integer_part
        #     # currency_part = pynutil.delete("currency: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"") + pynutil.delete(NEMO_SIGMA)
        #     # currency_part = pynini.compose(graph_decimal, currency_part)
        #     # # integer_part = pynutil.delete(NEMO_SIGMA) + pynini.accep("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynini.accep("\"") + pynutil.insert(currency_part) + NEMO_SIGMA
        #     # integer_part = pynutil.delete(NEMO_SIGMA) + pynini.accep("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynini.accep("\"") + pynutil.delete(NEMO_SIGMA)
        #
        #     # extract major currency and insert corresponding minor currency
        #     currency = pynutil.delete("currency: \"") + NEMO_SIGMA + pynutil.delete("\"") + pynutil.delete(NEMO_SIGMA)
        #     currency_maj = pynini.compose(graph_decimal, currency)
        #     currency_maj_sing_form = pynini.compose(currency_maj, PLURAL_TO_SINGULAR)
        #     currency_min_sing = pynini.compose(currency_maj_sing_form, minor_singular)
        #     currency_min_plural = pynini.compose(currency_maj_sing_form, minor_plural)
        #
        #     frac_one = (
        #         pynutil.delete(NEMO_SIGMA) + pynini.accep("fractional_part: \"one\"") + pynutil.delete(NEMO_SIGMA)
        #     )
        #     frac_non_one = (
        #         pynutil.delete(NEMO_SIGMA)
        #         + pynini.accep("fractional_part: \"")
        #         + pynini.difference(NEMO_SIGMA, "one")
        #         + pynini.accep("\"")
        #         + pynutil.delete(NEMO_SIGMA)
        #     )
        #
        #     integer_part = (
        #         pynutil.delete(NEMO_SIGMA) + pynini.accep("fractional_part: \"one\"") + pynutil.delete(NEMO_SIGMA)
        #     )
        #
        #     graph_decimal_with_minor_currency_plural = pynini.compose(graph_decimal, frac_non_one) + pynutil.insert(
        #         " currency_minor: \"" + currency_min_plural + "\"" + currency_maj_sing_form
        #     )
        #     graph_decimal_with_minor_currency_plural = pynini.compose(
        #         graph_decimal_with_minor_currency_plural,
        #         pynutil.delete("currency: \"")
        #         + pynutil.delete(pynini.closure(NEMO_NOT_QUOTE, 1))
        #         + pynutil.delete("\"")
        #         + NEMO_SIGMA,
        #     )
        #
        #     graph_decimal_with_minor_currency_singular = pynini.compose(graph_decimal, frac_one) + pynutil.insert(
        #         " currency_minor: \"" + currency_min_sing + pynutil.insert("\"")
        #     )
        #     final_graph |= graph_decimal_with_minor_currency_singular
        #     final_graph |= graph_decimal_with_minor_currency_plural
        #
        # # graph_with_added_min_currency = (
        # #     graph_unit_plural
        # #     + insert_space
        # #     + decimal.graph_integer
        # #     + pynutil.insert(" " + pynini.compose(graph_decimal, pynini.cdrewrite(NEMO_SIGMA + pynutil.delete("integer_part" + NEMO_SIGMA + "\""), "", "", NEMO_SIGMA)))
        # #     + pynini.cross(".", " point ")
        # #     + decimal.graph_fractional
        # #     + insert_space
        # #     + pynutil.insert(currency_min_plural)
        # # )
        #
        # unit_insert = pynini.cdrewrite(pynutil.insert(minor_plural), "[BOS]", "", NEMO_SIGMA)
        # graph_with_added_min_currency = (
        #         graph_unit_plural
        #         + insert_space
        #         + decimal.graph_integer
        #         + pynutil.delete(NEMO_SIGMA)
        # )

        # final_graph = graph_with_added_min_currency

        # "$5.2" -> "$5 cur_min: "cur_maj" .2"

        if not deterministic:
            currencies = load_labels(get_abs_path("data/currency/currency.tsv"))
            integer_graph = None
            decimal_graph_with_minor = None
            decimal_graph_default = None

            for curr_symbol, curr_name in currencies:
                graph_end = pynutil.insert(" currency: \"" + curr_symbol + "\"")
                preserve_order = pynutil.insert(" preserve_order: True")
                integer_graph_curr = pynutil.delete(curr_symbol) + decimal.graph_integer + graph_end + preserve_order
                decimal_graph_with_minor_curr = (
                    integer_graph_curr + pynini.cross(".", " ") + decimal.graph_fractional + graph_end
                )
                decimal_graph_default_curr = (
                    pynutil.delete("currency: \"" + curr_name + NEMO_SIGMA)
                    + pynini.accep("integer_part")
                    + NEMO_SIGMA
                    + pynutil.insert(" currency: \"" + pynini.compose(curr_symbol, unit_plural) + "\"")
                )
                decimal_graph_default_curr = pynini.compose(graph_decimal, decimal_graph_default_curr)

                integer_graph = (
                    integer_graph_curr if integer_graph is None else pynini.union(integer_graph, integer_graph_curr)
                )
                decimal_graph_with_minor = (
                    decimal_graph_with_minor_curr
                    if decimal_graph_with_minor is None
                    else pynini.union(decimal_graph_with_minor, decimal_graph_default_curr)
                )
                decimal_graph_default = (
                    decimal_graph_default_curr
                    if decimal_graph_default is None
                    else pynini.union(decimal_graph_default, decimal_graph_default_curr)
                )

            final_graph = decimal_graph_with_minor | decimal_graph_default | integer_graph

        # from pynini.lib.rewrite import top_rewrites
        # import pdb; pdb.set_trace()
        #     # print(top_rewrites("$5", integer_graph, 5))
        # print(top_rewrites("$5.3", final_graph, 5))
        #         print()
        #
        # graph = None
        # for cur_maj in ["dollars", "euro", "pound"]:
        #     graph_with_min = NEMO_SIGMA + pynini.cross("fractional_part", "currency: \"" + cur_maj + "\" fractional_part") + NEMO_SIGMA + pynutil.insert(" currency: \"" + cur_maj + "\"")
        #     if graph is None:
        #         graph = graph_with_min
        #     else:
        #         graph |= graph_with_min
        #
        # graph = pynini.compose(graph_decimal, graph)
        #
        #
        # final_graph = None
        # for cur_maj in ["dollars", "euro", "pound"]:
        #     # remove = pynini.compose(graph, pynini.cdrewrite(pynutil.delete(cur_maj) + NEMO_SIGMA + pynini.accep(cur_maj), NEMO_DIGIT, ".", NEMO_SIGMA))
        #     remove = pynini.compose(graph, pynutil.delete("currency: \"" + cur_maj + pynutil.delete("\"")) + NEMO_SIGMA + pynini.accep(cur_maj) + NEMO_SIGMA + pynutil.insert(" preserve_order: True"))
        #     if final_graph is None:
        #         final_graph = remove
        #     else:
        #         final_graph |= remove

        # remove = pynini.compose(graph, pynutil.delete("currency: \"" + cur_maj + pynutil.delete("\"")) + NEMO_SIGMA + pynini.accep(cur_maj) + NEMO_SIGMA)

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
