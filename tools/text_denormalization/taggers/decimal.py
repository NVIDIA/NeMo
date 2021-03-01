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

import pynini
from denormalization.data_loader_utils import get_abs_path
from denormalization.graph_utils import GraphFst, delete_extra_space, delete_space
from denormalization.taggers.cardinal import CardinalFst
from pynini.lib import pynutil


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, 
        e.g. minus twelve point five o o six -> decimal { negative: "true" integer_part: "12"  fractional_part: "5006" }
    """

    def __init__(self):
        super().__init__(name="decimal", kind="classify")
        # negative, fractional_part, quantity, exponent, style(depre)

        cardinal_graph = CardinalFst().graph_no_exception

        graph_decimal = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_decimal |= pynini.string_file(get_abs_path("data/numbers/zero.tsv")) | pynini.cross("o", "0")

        graph_decimal = pynini.closure(graph_decimal + delete_space) + graph_decimal
        self.graph = graph_decimal

        point = pynini.cross("point", "")

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus", "\"true\"") + delete_extra_space, 0, 1
        )

        graph_fractional = pynutil.insert("fractional_part: \"") + graph_decimal + pynutil.insert("\"")
        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        final_graph = (
            optional_graph_negative
            + pynini.closure(graph_integer + delete_extra_space, 0, 1)
            + point
            + delete_extra_space
            + graph_fractional
        )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
