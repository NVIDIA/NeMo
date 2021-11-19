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

from nemo_text_processing.inverse_text_normalization.vi.graph_utils import GraphFst, delete_space


try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class AddressFst(GraphFst):
    """
    Finite state transducer for classifying address
        e.g. hai sẹc ba -> tokens { address { value: "2/3" } }
        e.g. hai sẹc mười sẹc năm -> tokens { address{ value: "2/10/5" } }

    Args:
        cardinal: OrdinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="address", kind="classify")

        graph_cardinal = cardinal.graph_no_exception

        split_component = pynini.cross(pynini.union("sẹc", "sẹt"), "/")
        graph_address = pynini.closure(graph_cardinal + delete_space + split_component + delete_space, 1) + graph_cardinal
        graph = pynutil.insert("value: \"") + graph_address + pynutil.insert("\"")

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
