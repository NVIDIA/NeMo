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
from nemo_text_processing.text_normalization.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    delete_space,
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
    Finite state transducer for classifying money, suppletive aware, e.g. 
        "5руб." -> money {  integer_part: "пять" currency: "рублей" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.cardinal_numbers
        decimal_graph = decimal.final_graph

        unit_singular = pynini.string_file(get_abs_path("ru/data/currency/currency_singular.tsv"))
        unit_plural = pynini.string_file(get_abs_path("ru/data/currency/currency_plural.tsv"))

        optional_delimiter = pynini.closure(pynini.cross(" ", ""), 0, 1)
        graph_unit_singular = (
            optional_delimiter + pynutil.insert(" currency: \"") + unit_singular + pynutil.insert("\"")
        )
        graph_unit_plural = optional_delimiter + pynutil.insert(" currency: \"") + unit_plural + pynutil.insert("\"")

        singular_graph = (
            pynutil.insert("integer_part: \"") + pynini.cross("1", "один") + pynutil.insert("\"") + graph_unit_singular
        )

        graph_decimal = decimal_graph + graph_unit_plural

        graph_integer = (
            pynutil.insert("integer_part: \"")
            + ((NEMO_SIGMA - "1") @ cardinal_graph)
            + pynutil.insert("\"")
            + graph_unit_plural
        )

        graph_integer |= singular_graph
        tagger_graph = (graph_integer.optimize() | graph_decimal.optimize()).optimize()

        # verbalizer
        integer = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        unit = (
            pynutil.delete("currency:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        unit = delete_space + pynutil.insert(" ") + unit

        verbalizer_graph_cardinal = (integer + unit).optimize()

        integer = pynutil.delete("\"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        integer_part = pynutil.delete("integer_part: ") + integer
        fractional_part = pynutil.delete("fractional_part: ") + integer
        verbalizer_graph_decimal = (
            pynutil.delete('decimal { ')
            + integer_part
            + pynini.accep(" ")
            + fractional_part
            + pynutil.delete(" }")
            + unit
        )

        verbalizer_graph = (verbalizer_graph_cardinal | verbalizer_graph_decimal).optimize()

        self.final_graph = (tagger_graph @ verbalizer_graph).optimize()
        self.fst = self.add_tokens(self.final_graph).optimize()

        # from pynini.lib.rewrite import top_rewrites
        # import pdb; pdb.set_trace()
        # print(top_rewrites("2,5 руб.", tagger_graph, 5))
        # print(top_rewrites("2,5 руб.", self.final_graph, 5))
        # print(top_rewrites('decimal { integer_part: "второго целых" fractional_part: "пяти десятые" } currency: "рублях"', verbalizer_graph_decimal, 5))
        # print()
