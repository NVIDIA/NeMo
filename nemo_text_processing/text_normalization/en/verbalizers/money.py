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
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    NEMO_WHITE_SPACE,
    SINGULAR_TO_PLURAL,
    GraphFst,
    delete_space,
    get_abs_path,
    insert_space,
)

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):

    PYNINI_AVAILABLE = False


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { integer_part: "twelve" fractional_part: "o five" currency: "dollars" } -> twelve o five dollars

    Args:
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        unit = (
            pynutil.delete("currency:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        graph = decimal.numbers + delete_space + pynutil.insert(" ") + unit

        if not deterministic:
            # For non-deterministic case, the currency symbol was not changed in the tagger, so here we need to
            # create a transducer to replace the currency symbol with the correct spoken equivalent

            # the graph finds instances where the fractional part is '.01' - this is need to add singular case for
            # the minor currency
            fractional_non_one = (
                pynutil.delete("fractional_part: \"")
                + pynini.difference(pynini.closure(NEMO_NOT_QUOTE), pynini.union("oh one", "o one", "zero one", "one"))
                + pynutil.delete("\"")
            )
            preserve_order = pynutil.delete("preserve_order: True")

            # Create units graph for major and minor currencies in both singular and plural forms
            unit_major_sing = pynini.string_file(get_abs_path("data/currency/currency.tsv"))
            unit_major_plural = (
                pynutil.delete("currency: \"")
                + pynini.compose(unit_major_sing, SINGULAR_TO_PLURAL)
                + pynutil.delete("\"")
            )
            unit_major_sing = pynutil.delete("currency: \"") + unit_major_sing + pynutil.delete("\"")
            unit_minor_sing = pynini.string_file(get_abs_path("data/currency/currency_minor_singular.tsv"))
            unit_minor_sing = pynutil.delete("currency: \"") + unit_minor_sing + pynutil.delete("\"")
            unit_minor_plural = pynini.string_file(get_abs_path("data/currency/currency_minor_plural.tsv"))
            unit_minor_plural = pynutil.delete("currency: \"") + unit_minor_plural + pynutil.delete("\"")

            # for the integer part of the money graph find cases, when the integer part is one
            # this is need to add a singular currency value, e.g. `$1` -> `one dollar` not `one dollars`
            integer_one = pynini.compose(decimal.integer, pynini.accep("one"))

            # graph for integer values that are not `1`, we need to use plural currency form for such cases
            integer_not_one = pynini.compose(decimal.integer, pynini.difference(NEMO_SIGMA, pynini.accep("one")))
            graph_integer = integer_one + delete_space + insert_space + unit_major_sing + delete_space + preserve_order
            graph_integer |= (
                integer_not_one + delete_space + insert_space + unit_major_plural + delete_space + preserve_order
            )

            # find when the fractional part is equal to `.01` -> to use singular form of the minor currency
            fractional_part_sing = (
                delete_space
                + pynutil.delete("fractional_part: \"" + pynini.union("o ", "oh ", "zero "))
                + pynini.accep("one")
                + pynutil.delete("\"")
                + delete_space
                + insert_space
                + unit_minor_sing
            )

            # verbalize money values with .01 in the fractional part and use singular form of the minor currency
            # e.g. '$12.01' -> 'twelve dollars (and) one cent'
            graph_decimal_with_minor = (
                graph_integer
                + delete_space
                + insert_space
                + pynini.closure(pynutil.insert("and "), 0, 1)
                + fractional_part_sing
            )

            fractional_part_plural = (
                delete_space + fractional_non_one + delete_space + insert_space + unit_minor_plural
            )

            # verbalize money values with the fractional part not equal to '.01' and
            # use plural form of the minor currency
            # e.g. '$12.56' -> 'twelve dollars (and) fifty six cents'
            graph_decimal_with_minor |= (
                graph_integer
                + delete_space
                + insert_space
                + pynini.closure(pynutil.insert("and "), 0, 1)
                + fractional_part_plural
            )

            # handle cases when there is no integer part
            graph_decimal_with_minor |= fractional_part_sing | fractional_part_plural

            # to make sure no texts with remaining currency symbol bypass the verbalizer
            graph = pynini.compose(pynini.closure(NEMO_ALPHA | ":" | "\"" | "{" | "}" | "_" | NEMO_WHITE_SPACE), graph)
            graph |= graph_integer | graph_decimal_with_minor

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
