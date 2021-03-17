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

import pynini
from pynini.lib import pynutil
from ..graph_utils import GraphFst
from .punctuation import PunctuationFst
from .tokenize_and_classify import ClassifyFst


class ClassifyFst(GraphFst):
    """
    Final FST that tokenizes an entire sentence
        e.g. its twelve thirty now. -> tokens { name: "its" } tokens { time { hours: "12" minutes: "30" } } tokens { name: "now" } tokens { name: "." pause_length: "PAUSE_LONG phrase_break: true type: PUNCT" }
    """

    def __init__(self):
        super().__init__(name="tokenize_and_classify_final", kind="classify")

        classify = ClassifyFst().fst
        punct = PunctuationFst().fst
        token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
        token_plus_punct = (
            pynini.closure(punct + pynutil.insert(" ")) + token + pynini.closure(pynutil.insert(" ") + punct)
        )
        graph = token_plus_punct + pynini.closure(pynini.cross(pynini.closure(" ", 1), " ") + token_plus_punct)
        graph = (
            pynini.closure(pynutil.delete(pynini.closure(" ", 1)), 0, 1)
            + graph
            + pynini.closure(pynutil.delete(pynini.closure(" ", 1)), 0, 1)
        )
        self.fst = graph.optimize()
