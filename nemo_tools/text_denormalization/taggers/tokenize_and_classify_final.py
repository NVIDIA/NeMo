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
from nemo_tools.text_denormalization.graph_utils import GraphFst, delete_extra_space, delete_space
from nemo_tools.text_denormalization.taggers.punctuation import PunctuationFst
from nemo_tools.text_denormalization.taggers.tokenize_and_classify import ClassifyFst
from pynini.lib import pynutil


class ClassifyFinalFst(GraphFst):
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

        graph = token_plus_punct + pynini.closure(delete_extra_space + token_plus_punct)
        graph = delete_space + graph + delete_space

        self.fst = graph.optimize()
