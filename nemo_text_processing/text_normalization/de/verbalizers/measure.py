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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_preserve_order,
)

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measure, e.g.
        measure { cardinal { integer: "два килограма" } } -> "два килограма"
    
    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, decimal: GraphFst, cardinal: GraphFst, fraction: GraphFst, deterministic: bool):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)
        optional_sign = cardinal.optional_sign
        unit = pynutil.delete("units: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        delete_space = pynini.closure(pynutil.delete(" "), 0, 1)

        graph_decimal = (
            pynutil.delete("decimal { ") + optional_sign + decimal.numbers + delete_space + pynutil.delete(" }")
        )
        graph_cardinal = (
            pynutil.delete("cardinal { ") + optional_sign + cardinal.numbers + delete_space + pynutil.delete(" }")
        )

        graph_fraction = pynutil.delete("fraction { ") + fraction.graph + delete_space + pynutil.delete(" }")

        graph = (graph_cardinal | graph_decimal | graph_fraction) + pynini.accep(" ") + unit

        graph |= unit + delete_extra_space + (graph_cardinal | graph_decimal)
        graph += delete_preserve_order

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
