# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.inverse_text_normalization.de.graph_utils import (
    NEMO_CHAR,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
        e.g. ein halb -> tokens { fraction { numerator: "1" denominator: "2" } }
        e.g. eineinhalb -> tokens { fraction { integer_part: "1" numerator: "1" denominator: "2" } }
        e.g. drei zwei hundertstel -> tokens { fraction { integer_part: "3" numerator: "2" denominator: "100" } }
    
    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="fraction", kind="classify")
        # integer_part # numerator # denominator

        cardinal_graph = cardinal.graph_no_exception
        fractional = pynini.string_file(get_abs_path("data/fractions.tsv"))

        self.fractional = ((pynini.closure(NEMO_CHAR) + fractional) @ cardinal_graph).optimize()

        integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        numerator = pynutil.insert("numerator: \"") + cardinal_graph + pynutil.insert("\"")
        denominator = pynutil.insert("denominator: \"") + self.fractional + pynutil.insert("\"")

        graph = pynini.closure(integer + delete_space, 0, 1) + numerator + delete_space + insert_space + denominator
        graph = graph.optimize()
        self.final_graph_wo_negative = graph

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus", "\"true\"") + delete_extra_space, 0, 1
        )
        graph = optional_graph_negative + graph
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
