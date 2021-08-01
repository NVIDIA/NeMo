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

        if not deterministic:
            currencies = load_labels(get_abs_path("data/currency/currency.tsv"))
            integer_graph = None
            decimal_graph_with_minor = None
            decimal_graph_default = None

            for curr_symbol, curr_name in currencies:
                curr_symbol_graph = pynutil.delete(curr_symbol)
                graph_end = pynutil.insert(" currency: \"" + curr_symbol + "\"")
                preserve_order = pynutil.insert(" preserve_order: True")
                integer_part = decimal.graph_integer + graph_end + preserve_order
                integer_graph_curr = curr_symbol_graph + integer_part
                decimal_graph_with_minor_curr = (
                    curr_symbol_graph
                    + pynini.closure(integer_part, 0, 1)
                    + pynini.cross(".", " ")
                    + decimal.graph_fractional
                    + graph_end
                )
                decimal_graph_default_curr = (
                    pynutil.delete("currency: \"" + curr_name + NEMO_SIGMA)
                    + pynini.closure(pynini.accep("integer_part") + NEMO_SIGMA, 0, 1)
                    + pynini.accep("fractional_part")
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

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

        # from pynini.lib.rewrite import top_rewrites
        #
        # import pdb;
        # pdb.set_trace()
        # print(top_rewrites("$.5", decimal_graph_default, 5))
        # print(top_rewrites("₩4 billion", final_graph, 5))
