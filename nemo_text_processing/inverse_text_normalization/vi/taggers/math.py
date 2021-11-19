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

from nemo_text_processing.inverse_text_normalization.vi.utils import get_abs_path
from nemo_text_processing.inverse_text_normalization.vi.graph_utils import (
    GraphFst,
    delete_space,
    delete_extra_space,
    NEMO_ALPHA
)


try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class MathFst(GraphFst):
    """
    Finite state transducer for classifying math equation
        e.g. x cộng y cộng z -> tokens { math { equation: "x + y + z" } }
        e.g. hai bình phương cộng trừ năm -> tokens { math { equation: "2² + -5" } }

    Args:
        cardinal: OrdinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="math", kind="classify")
        # integer_part # numerator # denominator

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_symbols = pynini.string_file(get_abs_path("data/math/symbols.tsv")).invert()
        graph_one = pynini.cross("mốt", "1")
        graph_four = pynini.cross("tư", "4")
        graph_five = pynini.cross("lăm", "5")

        graph_cardinal = cardinal.graph_no_exception
        optional_graph_negative = pynini.closure(pynini.cross(pynini.union("âm", "trừ"), "-") + delete_space, 0, 1)
        optional_graph_power = pynini.closure(
            delete_space + pynini.cross("bình phương", "²") | delete_space + pynini.cross("lập phương", "³"), 0, 1
        )

        graph_digit = graph_digit | graph_zero
        graph_fraction = pynini.union(
            graph_digit,
            graph_four,
            pynini.closure(graph_digit + delete_space, 1) + (graph_digit | graph_four | graph_five | graph_one),
        )
        optional_graph_fraction = pynini.closure(
            delete_space + pynini.cross(pynini.union("chấm", "phẩy"), ".") + delete_space + graph_fraction, 0, 1
        )
        graph_decimal = graph_cardinal + optional_graph_fraction

        alpha_num = NEMO_ALPHA | graph_decimal
        graph_equation = (
            pynini.closure(
                optional_graph_negative
                + alpha_num
                + optional_graph_power
                + delete_extra_space
                + graph_symbols
                + delete_extra_space, 1
            )
            + optional_graph_negative
            + alpha_num)

        graph = pynutil.insert("equation: \"") + graph_equation + pynutil.insert("\"")
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
