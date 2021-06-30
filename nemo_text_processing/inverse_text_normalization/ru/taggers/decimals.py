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


from nemo_text_processing.text_normalization.graph_utils import GraphFst, delete_extra_space
from nemo_text_processing.text_normalization.ru.taggers.cardinal import CardinalFst as CardinalFstTN
from nemo_text_processing.text_normalization.ru.taggers.decimals import DecimalFst as DecimalFstTN
from nemo_text_processing.text_normalization.ru.taggers.ordinal import OrdinalFst as OrdinalFstTN

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal
        e.g. "минус две целых пять десятых" -> negative: "true" integer_part: "2," fractional_part: "5"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = False):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("минус", "\"true\"") + delete_extra_space, 0, 1
        )

        graph_fractional_tn = DecimalFstTN(cardinal=CardinalFstTN(), ordinal=OrdinalFstTN())
        graph_fractional_part = pynini.invert(graph_fractional_tn.graph_fractional).optimize()
        graph_integer_part = pynini.invert(graph_fractional_tn.integer_part).optimize()

        graph_fractional = pynutil.insert("fractional_part: \"") + graph_fractional_part + pynutil.insert("\"")
        graph_integer = pynutil.insert("integer_part: \"") + graph_integer_part + pynutil.insert("\"")
        self.final_graph_wo_sign = graph_integer + pynini.accep(" ") + graph_fractional
        final_graph = optional_graph_negative + self.final_graph_wo_sign

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
