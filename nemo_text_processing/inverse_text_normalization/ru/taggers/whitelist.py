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


from nemo_text_processing.text_normalization.en.graph_utils import GraphFst

try:
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelist, e.g.
        "квартира" -> telephone { number_part: "кв." }

    Args:
        tn_whitelist: Text normalization whitelist graph
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, tn_whitelist: GraphFst, deterministic: bool = True):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)

        tn_whitelist = tn_whitelist.final_graph
        graph = tn_whitelist.invert().optimize()
        graph = pynutil.insert("name: \"") + graph + pynutil.insert("\"")
        self.fst = graph.optimize()
