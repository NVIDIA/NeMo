# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
)
from nemo_text_processing.text_normalization.sv.utils import get_abs_path
from pynini.lib import pynutil


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    "23 4/5" ->
    tokens { fraction { integer: "tjugotre" numerator: "fyra" denominator: "fem" } }
    # en åttondel (1/8)

    Args:
        cardinal: CardinalFst
        ordinal: OrdinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph
        ordinal_graph = ordinal.graph
        numerator_graph = cardinal.graph_en

        fractional_endings = pynini.string_map([
            ("ljarte", "ljarddel"),
            ("tionde", "tiondel"),
            ("tonde", "tondel"),
            ("ljonte", "ljondel"),
            ("lliarte", "lliarddel"),
            ("llionte", "lliondel"),
            ("tusende", "tusendel"),
            ("te", "tedel"),
            ("de", "dedel"),
            ("je", "jedel"),
            ("drade", "dradel"),
            ("a", "adel")
        ])
        alt_fractional_endings = pynini.string_map([
            ("tondel", "tondedel"),
            ("tiondel", "tiondedel")
        ])
        lexicalised = pynini.string_map([
            ("andradel", "halv"),
            ("fjärdedel", "kvart")
        ])
        alt_lexicalised = pynini.string_map([
            ("halv", "andradel"),
            ("kvart", "fjärdedel"),
            ("kvart", "kvarts")
        ])

        fractions = (
            ordinal_graph
            @ pynini.cdrewrite(fractional_endings, "", "[EOS]", NEMO_SIGMA)
            @ pynini.cdrewrite(lexicalised, "[BOS]", "[EOS]", NEMO_SIGMA)
        )
        fractions_alt = (
            fractions
            @ pynini.cdrewrite(alt_fractional_endings, "", "[EOS]", NEMO_SIGMA)
            @ pynini.cdrewrite(alt_lexicalised, "[BOS]", "[EOS]", NEMO_SIGMA)
        )
        if not deterministic:
            fractions |= fractions_alt

        self.fractions = fractions

        integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"")
        numerator = (
            pynutil.insert("numerator: \"") + numerator_graph + (pynini.cross("/", "\" ") | pynini.cross(" / ", "\" "))
        )

        denominator = pynutil.insert("denominator: \"") + fractions + pynutil.insert("\"")

        graph = pynini.closure(integer + pynini.accep(" "), 0, 1) + (numerator + denominator)
        graph |= pynini.closure(integer + (pynini.accep(" ") | pynutil.insert(" ")), 0, 1) + pynini.compose(
            pynini.string_file(get_abs_path("data/numbers/fraction.tsv")), (numerator + denominator)
        )

        self.graph = graph
        final_graph = self.add_tokens(self.graph)
        self.fst = final_graph.optimize()
