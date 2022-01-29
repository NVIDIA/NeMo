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


from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_CHAR,
    NEMO_SIGMA,
    GraphFst,
    insert_space,
)
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

    def __init__(self, deterministic: bool = True):
        super().__init__(name="roman", kind="classify", deterministic=deterministic)

        default_graph = pynini.string_file(get_abs_path("data/roman/roman_to_spoken.tsv")).optimize()
        default_graph = pynutil.insert("integer: \"") + default_graph + pynutil.insert("\"")
        names = pynini.string_map(load_labels(get_abs_path("data/roman/male.tsv"))).optimize()
        names |= pynini.string_map(load_labels(get_abs_path("data/roman/female.tsv"))).optimize()

        # roman numerals from I to IV with a preceding name are converted to ordinal form
        graph = (
            pynutil.insert("key_the_ordinal: \"")
            + names
            + pynutil.insert("\"")
            + pynini.accep(" ")
            + pynini.compose(pynini.union("I", "II", "III", "IV"), default_graph)
        ).optimize()
        key_words = pynini.string_map(load_labels(get_abs_path("data/roman/key_words.tsv"))).optimize()

        # single symbol roman numerals with preceding key words are converted to cardinal form
        graph = pynini.leniently_compose(
            graph,
            pynutil.insert("key_cardinal: \"")
            + key_words
            + pynutil.insert("\"")
            + pynini.accep(" ")
            + pynini.compose(NEMO_ALPHA, graph),
            NEMO_SIGMA,
        ).optimize()

        # two and more roman numerals, single digit roman numbers could be initials or I
        roman_to_cardinal = pynini.compose(
            pynini.closure(NEMO_ALPHA, 2), (pynutil.insert("default_cardinal: \"default\" ") + default_graph)
        )

        graph |= roman_to_cardinal
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
