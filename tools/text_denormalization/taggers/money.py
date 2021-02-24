# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from denormalization.graph_utils import GraphFst, convert_space, delete_extra_space, get_plurals
from denormalization.taggers.cardinal import CardinalFst
from denormalization.taggers.decimal import DecimalFst
from pynini.lib import pynutil


class MoneyFst(GraphFst):
    def __init__(self):
        super().__init__(name="money", kind="classify")
        # amount: Decimal, quantity, currency, style(depr)
        cardinal_graph = CardinalFst().graph_no_exception
        graph_decimal = DecimalFst().graph

        graph_unit = pynini.string_file(get_abs_path("data/currency.tsv"))
        graph_unit = pynini.invert(graph_unit)
        graph_unit = get_plurals(graph_unit)

        point = pynini.cross("point", "")

        graph_fractional = pynutil.insert("fractional_part: \"") + graph_decimal + pynutil.insert("\"")
        graph_integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        unit = pynutil.insert("currency: \"") + convert_space(graph_unit) + pynutil.insert("\"")

        graph_decimal = (
            pynutil.insert("amount { ")
            + pynini.union(
                pynini.closure(graph_integer + delete_extra_space, 0, 1)
                + point
                + delete_extra_space
                + graph_fractional,
                graph_integer,
            )
            + pynutil.insert(" }")
        )
        final_graph = graph_decimal + delete_extra_space + unit
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
