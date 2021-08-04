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
            fractional_non_one = (
                pynutil.delete("fractional_part: \"")
                + pynini.difference(pynini.closure(NEMO_NOT_QUOTE), pynini.union("oh one", "o one", "zero one", "one"))
                + pynutil.delete("\"")
            )
            preserve_order = pynutil.delete("preserve_order: True")
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

            integer_one = pynini.compose(decimal.integer, pynini.accep("one"))
            integer_not_one = pynini.compose(decimal.integer, pynini.difference(NEMO_SIGMA, pynini.accep("one")))
            graph_integer = integer_one + delete_space + insert_space + unit_major_sing + delete_space + preserve_order
            graph_integer |= (
                integer_not_one + delete_space + insert_space + unit_major_plural + delete_space + preserve_order
            )
            fractional_part_sing = (
                delete_space
                + pynutil.delete("fractional_part: \"" + pynini.union("o ", "oh ", "zero "))
                + pynini.accep("one")
                + pynutil.delete("\"")
                + delete_space
                + insert_space
                + unit_minor_sing
            )
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
            graph_decimal_with_minor |= (
                graph_integer
                + delete_space
                + insert_space
                + pynini.closure(pynutil.insert("and "), 0, 1)
                + fractional_part_plural
            )
            graph_decimal_with_minor |= fractional_part_sing | fractional_part_plural

            # to make sure no texts with remaining currency symbol bypass the verbalizer
            graph = pynini.compose(pynini.closure(NEMO_ALPHA | ":" | "\"" | "{" | "}" | "_" | NEMO_WHITE_SPACE), graph)
            graph |= graph_integer | graph_decimal_with_minor

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
