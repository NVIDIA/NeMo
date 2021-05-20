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

from nemo_text_processing.text_normalization.graph_utils import GraphFst, delete_space

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class SerialFst(GraphFst):
    """
    Finite state transducer for verbalizing serial, e.g.
        tokens { serial { value: "c thirty two five" } } -> c thirty two five

    Args:
        measure: MeasureFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, measure: GraphFst, deterministic: bool = False):
        super().__init__(name="serial", kind="verbalize", deterministic=deterministic)

        serial = pynutil.delete("units: \"") + pynini.cross("serial", "") + pynutil.delete("\"") + delete_space
        graph = measure.graph_cardinal + delete_space + serial
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
