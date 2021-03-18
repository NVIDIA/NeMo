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
from nemo_tools.text_denormalization.graph_utils import NEMO_CHAR, GraphFst
from nemo_tools.text_denormalization.taggers.cardinal import CardinalFst
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        e.g. thirteenth -> ordinal { integer: "13" }
    """

    def __init__(self):
        super().__init__(name="ordinal", kind="classify")

        cardinal_graph = CardinalFst().graph_no_exception

        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv"))
        graph_teens = pynini.string_file(get_abs_path("data/ordinals/teen.tsv"))
        # change to General UTF8
        graph = pynini.closure(NEMO_CHAR) + pynini.union(
            graph_digit, graph_teens, pynini.cross("tieth", "ty"), pynini.cross("th", "")
        )

        self.graph = graph @ cardinal_graph
        final_graph = pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
