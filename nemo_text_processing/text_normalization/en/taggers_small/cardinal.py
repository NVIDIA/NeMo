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


from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, GraphFst
from nemo_text_processing.text_normalization.en.taggers.cardinal import CardinalFst as defaultCardinalFst

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g. 
        -23 -> cardinal { negative: "true"  integer: "twenty three" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        default_cardinals = defaultCardinalFst()

        filter = NEMO_DIGIT ** (5, ...)
        final_graph = pynini.compose(filter, default_cardinals.final_graph).optimize()

        final_graph = (
            default_cardinals.optional_minus_graph + pynutil.insert("integer: \"") + final_graph + pynutil.insert("\"")
        )
        final_graph = self.add_tokens(final_graph.optimize())
        self.fst = final_graph.optimize()
