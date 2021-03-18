# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
from nemo_tools.text_denormalization.data_loader_utils import get_abs_path
from nemo_tools.text_denormalization.graph_utils import NEMO_DIGIT, GraphFst, delete_extra_space, delete_space
from nemo_tools.text_denormalization.taggers.cardinal import CardinalFst
from pynini.lib import pynutil

cardinal = CardinalFst()
cardinal_graph = cardinal.graph_no_exception


def get_quantity(deci):
    numbers = cardinal.graph_hundred_component_at_least_one_none_zero_digit @ (
        pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT)
    )
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
    res |= deci + delete_extra_space + pynutil.insert("quantity: \"") + (suffix | "thousand") + pynutil.insert("\"")
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, 
        e.g. minus twelve point five o o six billion -> decimal { negative: "true" integer_part: "12"  fractional_part: "5006" quantity: "billion" }
    """

    def __init__(self):
        super().__init__(name="decimal", kind="classify")
        # negative, fractional_part, quantity, exponent, style(depre)

        graph_decimal = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_decimal |= pynini.string_file(get_abs_path("data/numbers/zero.tsv")) | pynini.cross("o", "0")

        graph_decimal = pynini.closure(graph_decimal + delete_space) + graph_decimal
        self.graph = graph_decimal

        point = pynutil.delete("point")

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus", "\"true\"") + delete_extra_space, 0, 1
        )

        graph_fractional = pynutil.insert("fractional_part: \"") + graph_decimal + pynutil.insert("\"")
        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        final_graph_wo_sign = (
            pynini.closure(graph_integer + delete_extra_space, 0, 1) + point + delete_extra_space + graph_fractional
        )
        final_graph = optional_graph_negative + final_graph_wo_sign

        self.final_graph_wo_negative = final_graph_wo_sign | get_quantity(final_graph_wo_sign)
        final_graph |= optional_graph_negative + get_quantity(final_graph_wo_sign)
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
