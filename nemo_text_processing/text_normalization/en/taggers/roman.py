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


from nemo_text_processing.text_normalization.en.graph_utils import NEMO_ALPHA, GraphFst, insert_space
from nemo_text_processing.text_normalization.en.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.en.utils import get_abs_path, load_labels

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class RomanFst(GraphFst):
    """
    Finite state transducer for classifying roman numbers:
        e.g. "IV" -> tokens { roman { integer: "four" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="roman", kind="classify", deterministic=deterministic)

        def _load_roman(file: str, lm: bool = False):
            roman = load_labels(get_abs_path(file))
            roman_numerals = [(x.upper(), y) for x, y in roman]
            if not lm:
                roman_numerals += [(x, y) for x, y in roman]
            return pynini.string_map(roman_numerals)

        cardinal_graph = CardinalFst(deterministic=True).graph
        digit_teen = _load_roman("data/roman/digit_teen.tsv", lm) @ cardinal_graph
        ties = _load_roman("data/roman/ties.tsv", lm) @ cardinal_graph
        hundreds = _load_roman("data/roman/hundreds.tsv", lm) @ cardinal_graph

        graph = (
            (ties | digit_teen | hundreds)
            | (ties + insert_space + digit_teen)
            | (hundreds + pynini.closure(insert_space + ties, 0, 1) + pynini.closure(insert_space + digit_teen, 0, 1))
        ).optimize()

        graph = graph + pynini.closure(pynutil.delete("."), 0, 1)

        # A single letter roman numbers are often confused with "I" or initials, add a little weight to make sure
        # the original token is among non-deterministic options
        graph = pynutil.add_weight(pynini.compose(NEMO_ALPHA, graph), 1) | (
            pynini.compose(pynini.closure(NEMO_ALPHA, 2), graph)
        )
        graph = pynutil.insert("integer: \"") + graph + pynutil.insert("\"")
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
