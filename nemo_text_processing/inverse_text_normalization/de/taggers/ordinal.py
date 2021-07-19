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

from nemo_text_processing.inverse_text_normalization.de.graph_utils import NEMO_CHAR, GraphFst
from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PYNINI_AVAILABLE = False


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        e.g. dreizehnter -> ordinal { integer: "13" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/ordinals/ties.tsv"))
        graph_thousands = pynini.string_file(get_abs_path("data/ordinals/thousands.tsv"))

        suffixes = pynini.union("ten", "tem", "ter", "tes", "te")

        self.graph = (
            (
                pynini.closure(NEMO_CHAR)
                + pynini.closure(pynini.union(graph_digit, graph_thousands, graph_ties), 0, 1)
                + pynutil.delete(suffixes)
            )
            @ cardinal_graph
        ).optimize()

        graph = pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        self.graph = self.graph.optimize()
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
