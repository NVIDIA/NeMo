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

from nemo_text_processing.text_normalization.en.graph_utils import (
    MIN_NEG_WEIGHT,
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    GraphFst,
    convert_space,
)
from nemo_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst

try:
    import pynini
    from pynini.lib import pynutil
    from pynini.examples import plurals

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class WordFst(GraphFst):
    """
    Finite state transducer for classifying word. Considers sentence boundary exceptions.
        e.g. sleep -> tokens { name: "sleep" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="word", kind="classify", deterministic=deterministic)

        punct = PunctuationFst().graph
        punct_symbols = punct.project("input")
        graph = pynini.closure(pynini.difference(NEMO_NOT_SPACE, punct_symbols), 1)

        if not deterministic:
            graph = pynini.closure(
                pynini.difference(
                    graph, pynini.union("$", "€", "₩", "£", "¥", "#", "$", "%") + pynini.closure(NEMO_DIGIT, 1)
                ),
                1,
            )

        # to allow alpha with punctuation words "I'm", "O'Neil", "Then..." to be tagged as a word
        # so that no extra spaces are added around punctuation mark
        # non_digit is needed to allow non-ascii chars, like in "Müller's"
        non_digit = pynini.difference(NEMO_NOT_SPACE, NEMO_DIGIT).optimize()
        at_least_one_alpha = (
            pynini.closure(non_digit) + pynini.closure(NEMO_ALPHA, 1) + pynini.closure(non_digit)
        ).optimize()

        # punct followed by word and another punct mark: { "And, }
        alpha_with_punct_graph = (
            pynini.closure(punct_symbols)
            + at_least_one_alpha
            + pynini.closure(punct_symbols, 1)
            + pynini.closure(non_digit)
            + pynini.closure(NEMO_ALPHA)
        ).optimize()

        # punct followed by word: { "And }
        alpha_with_punct_graph |= pynini.closure(punct_symbols) + at_least_one_alpha
        alpha_with_punct_graph = pynutil.add_weight(alpha_with_punct_graph.optimize(), MIN_NEG_WEIGHT).optimize()
        graph |= alpha_with_punct_graph

        # leave phones of format [HH AH0 L OW1] untouched
        phoneme_unit = pynini.closure(NEMO_ALPHA, 1) + pynini.closure(NEMO_DIGIT)
        phoneme = (
            pynini.accep(pynini.escape("["))
            + pynini.closure(phoneme_unit + pynini.accep(" "))
            + phoneme_unit
            + pynini.accep(pynini.escape("]"))
        )

        if not deterministic:
            phoneme = (
                pynini.accep(pynini.escape("["))
                + pynini.closure(pynini.accep(" "), 0, 1)
                + pynini.closure(phoneme_unit + pynini.accep(" "))
                + phoneme_unit
                + pynini.closure(pynini.accep(" "), 0, 1)
                + pynini.accep(pynini.escape("]"))
            )

        self.graph = plurals._priority_union(convert_space(phoneme), graph, NEMO_SIGMA)
        self.fst = (pynutil.insert("name: \"") + self.graph + pynutil.insert("\"")).optimize()
