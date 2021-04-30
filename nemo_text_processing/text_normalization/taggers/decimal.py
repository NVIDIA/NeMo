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
from nemo_text_processing.text_normalization.graph_utils import GraphFst, delete_extra_space, delete_space

try:
    import pynini
    from pynini.lib import pynutil

    delete_space = pynutil.delete(" ")

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


def get_quantity(decimal: 'pynini.FstLike', cardinal_up_to_hundred: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. 1 million -> integer_part: "one" quantity: "million"
    e.g. 1.5 million -> integer_part: "one" fractional_part: "five" quantity: "million"

    Args: 
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
    """
    numbers = cardinal_up_to_hundred
    suffix = pynini.union("million", "billion", "trillion", "quadrillion", "quintillion", "sextillion")
    res = (
        pynutil.insert("integer_part: \"")
        + numbers
        + pynutil.insert("\"")
        + delete_extra_space
        + pynutil.insert("quantity: \"")
        + suffix
        + pynutil.insert("\"")
    )
    res |= decimal + delete_extra_space + pynutil.insert("quantity: \"") + (suffix | "thousand") + pynutil.insert("\"")
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g. 
        -12.5006 billion -> decimal { negative: "true" integer_part: "12"  fractional_part: "five o o six" quantity: "billion" }
        1 billion -> decimal { integer_part: "one" quantity: "billion" }

    cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")

        cardinal_graph = cardinal.graph
        cardinal_graph_hundred_component_at_least_one_none_zero_digit = (
            cardinal.graph_hundred_component_at_least_one_none_zero_digit
        )

        graph_decimal = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_decimal |= pynini.string_file(get_abs_path("data/numbers/zero.tsv"))

        graph_decimal = (
            pynini.cross("zero", "0")
            | graph_decimal
            | (graph_decimal | pynini.cross("o", "0"))
            + pynini.closure(delete_space + (graph_decimal | pynini.cross("o", "0")), 1)
        )
        self.graph = pynini.invert(graph_decimal).optimize()

        point = pynutil.delete(".")

        optional_graph_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        graph_fractional = pynutil.insert("fractional_part: \"") + self.graph + pynutil.insert("\"")
        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        final_graph_wo_sign = (
            pynini.closure(graph_integer + pynutil.insert(" "), 0, 1) + point + pynutil.insert(" ") + graph_fractional
        )

        self.final_graph_wo_negative = final_graph_wo_sign | get_quantity(
            final_graph_wo_sign, cardinal_graph_hundred_component_at_least_one_none_zero_digit
        )

        final_graph = optional_graph_negative + self.final_graph_wo_negative

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
