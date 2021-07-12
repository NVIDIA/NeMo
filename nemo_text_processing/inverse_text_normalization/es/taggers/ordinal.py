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

from nemo_text_processing.inverse_text_normalization.utils import get_abs_path
from nemo_text_processing.text_normalization.graph_utils import NEMO_CHAR, GraphFst, delete_space

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PYNINI_AVAILABLE = False


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        e.g. thirteenth -> ordinal { integer: "13" }

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="ordinal", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        graph_digit = pynini.string_file(get_abs_path("es/data/ordinals/digit.tsv"))
        graph_teens = pynini.string_file(get_abs_path("es/data/ordinals/teen.tsv"))
        graph_twenties = pynini.string_file(get_abs_path("es/data/ordinals/twenties.tsv"))
        graph_ties = pynini.string_file(get_abs_path("es/data/ordinals/ties.tsv"))
        graph_hundreds = pynini.string_file(get_abs_path("es/data/ordinals/hundreds.tsv"))

        ordinal_graph_union = pynini.union(graph_digit, graph_teens, graph_twenties, graph_ties, graph_hundreds,)

        accept_o_endings = NEMO_CHAR.closure() + pynini.accep("o")
        accept_a_endings = NEMO_CHAR.closure() + pynini.accep("a")
        accept_er_endings = NEMO_CHAR.closure() + pynini.accep("er")

        ordinal_graph_o = accept_o_endings @ ordinal_graph_union
        ordinal_graph_a = accept_a_endings @ ordinal_graph_union
        ordinal_graph_er = accept_er_endings @ ordinal_graph_union

        # 'optional_numbers_in_front' with negative weight so we always
        # include them if they're there
        optional_numbers_in_front = (pynutil.add_weight(ordinal_graph_union, -0.1) + delete_space.closure()).closure()

        graph_o_suffix = (optional_numbers_in_front + ordinal_graph_o) @ cardinal_graph

        graph_a_suffix = (optional_numbers_in_front + ordinal_graph_a) @ cardinal_graph

        graph_er_suffix = (optional_numbers_in_front + ordinal_graph_er) @ cardinal_graph

        graph = (
            pynutil.insert("integer: \"") + graph_o_suffix + pynutil.insert("\"") + pynutil.insert(" suffix: \"o\"")
        )
        graph |= (
            pynutil.insert("integer: \"") + graph_a_suffix + pynutil.insert("\"") + pynutil.insert(" suffix: \"a\"")
        )
        graph |= (
            pynutil.insert("integer: \"") + graph_er_suffix + pynutil.insert("\"") + pynutil.insert(" suffix: \"er\"")
        )

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
