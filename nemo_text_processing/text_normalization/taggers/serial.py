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

from nemo_text_processing.text_normalization.graph_utils import NEMO_ALPHA, GraphFst, delete_space

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class SerialFst(GraphFst):
    """
    Finite state transducer for classifying serial.
        The serial is a combination of digits, letters and dashes, e.g.:
        c325 ->
        tokens { serial { value: "c three hundred twenty five" } }
        tokens { serial { value: "c three two five" } }
        tokens { serial { value: "c thirty two five" } }
        tokens { serial { value: "c three twenty five" } }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = False):
        super().__init__(name="serial", kind="classify", deterministic=deterministic)

        num_graph = cardinal.graph
        serial_graph_cardinal_start = (
            pynini.closure((NEMO_ALPHA + pynutil.insert(" ")) | (NEMO_ALPHA + pynini.cross('-', ' ')), 1) + num_graph
        )
        serial_end = pynini.closure(pynutil.insert(" ") + NEMO_ALPHA + pynini.closure(pynutil.insert(" ") + num_graph))

        serial_graph_cardinal_end = num_graph + (
            (pynutil.insert(" ") + NEMO_ALPHA) | (pynini.cross('-', ' ') + NEMO_ALPHA)
        )
        serial_end2 = pynini.closure(
            pynutil.insert(" ")
            + num_graph
            + pynini.closure((pynutil.insert(" ") | pynini.cross("-", " ")) + NEMO_ALPHA)
        )

        serial_graph = (serial_graph_cardinal_start | serial_graph_cardinal_end) + pynini.closure(
            serial_end | serial_end2
        )

        graph = (
            pynutil.insert("cardinal { integer: \"")
            + serial_graph
            + delete_space
            + pynutil.insert("\" } units: \"serial\"")
        )

        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
