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
    NEMO_CHAR,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    GraphFst,
    delete_extra_space,
    delete_space,
)
from nemo_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.en.verbalizers.verbalize import VerbalizeFst
from nemo_text_processing.text_normalization.en.verbalizers.word import WordFst

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class VerbalizeFinalFst(GraphFst):
    """
    Finite state transducer that verbalizes an entire sentence, e.g.
    tokens { name: "its" } tokens { time { hours: "twelve" minutes: "thirty" } } tokens { name: "now" } tokens { name: "." } -> its twelve thirty now .

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        punct_post_process: if True punctuation will be normalized after the default normalization is complete
    """

    def __init__(self, deterministic: bool = True, punct_post_process: bool = True):
        super().__init__(name="verbalize_final", kind="verbalize", deterministic=deterministic)
        verbalize = VerbalizeFst(deterministic=deterministic).fst
        word = WordFst(deterministic=deterministic).fst
        types = verbalize | word

        if deterministic:
            graph = (
                pynutil.delete("tokens")
                + delete_space
                + pynutil.delete("{")
                + delete_space
                + types
                + delete_space
                + pynutil.delete("}")
            )
        else:
            graph = delete_space + types + delete_space

        graph = delete_space + pynini.closure(graph + delete_extra_space) + graph + delete_space

        if punct_post_process:
            punct_graph = self.punct_postprocess_graph()
            graph = pynini.compose(graph, punct_graph).optimize()

        self.fst = graph

    def punct_postprocess_graph(self):
        punct_marks_all = PunctuationFst().punct_marks

        # no_space_before_punct assume no space before them
        quotes = ["'", "\"", "``", "«"]
        dashes = ["-", "—"]
        open_close_symbols = {"<": ">", "{": "}", '"': '"', "``": "``", "``": "``", "(": ")", "“": "”"}  # , "'": "'"}
        allow_space_before_punct = ["&"] + quotes + dashes + [str(k) for k in open_close_symbols.keys()]
        no_space_before_punct = [m for m in punct_marks_all if m not in allow_space_before_punct]
        no_space_before_punct = pynini.union(*no_space_before_punct)
        delete_space = pynutil.delete(" ")

        # non_punct allows space
        # delete space before no_space_before_punct marks, if present
        non_punct = pynini.difference(NEMO_CHAR, no_space_before_punct).optimize()
        graph = (
            pynini.closure(non_punct)
            + pynini.closure(
                no_space_before_punct | pynutil.add_weight(delete_space + no_space_before_punct, MIN_NEG_WEIGHT)
            )
            + pynini.closure(non_punct)
        )
        graph = pynini.closure(graph).optimize()

        open_close_marks_graph = (
            pynini.accep("(")
            + pynini.closure(delete_space, 0, 1)
            + NEMO_NOT_SPACE
            + NEMO_SIGMA
            + pynini.closure(delete_space, 0, 1)
            + pynini.accep(")")
        )
        for open, close in open_close_symbols.items():
            open_close_marks_graph |= (
                pynini.accep(open)
                + pynini.closure(delete_space, 0, 1)
                + NEMO_NOT_SPACE
                + NEMO_SIGMA
                + pynini.closure(delete_space, 0, 1)
                + pynini.accep(close)
            )

        open_close_marks_graph = pynutil.add_weight(open_close_marks_graph, MIN_NEG_WEIGHT)
        open_close_marks_graph = NEMO_SIGMA + pynini.closure(open_close_marks_graph + NEMO_SIGMA)
        graph = pynini.compose(graph, open_close_marks_graph).optimize()
        return graph
