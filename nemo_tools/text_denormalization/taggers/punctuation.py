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
from nemo_tools.text_denormalization.graph_utils import GraphFst, convert_space
from pynini.lib import pynutil


class PunctuationFst(GraphFst):
    """
    Finite state transducer for classifying punctuation
        e.g. a, -> tokens { name: "a" } tokens { name: "," pause_length: "PAUSE_MEDIUM phrase_break: true type: PUNCT" }
    """

    def __init__(self):
        super().__init__(name="punctuation", kind="classify")

        medium_punct = pynini.union(",", ";", "(", ")")
        long_punct = pynini.union(".", "!", "?", ":")

        medium = (
            pynutil.insert("tokens { name: \"")
            + convert_space(medium_punct)
            + pynutil.insert("\"")
            + pynutil.insert(" pause_length: \"")
            + convert_space(pynutil.insert("PAUSE_MEDIUM phrase_break: true type: PUNCT"))
            + pynutil.insert("\" }")
        )
        loong = (
            pynutil.insert("tokens { name: \"")
            + convert_space(long_punct)
            + pynutil.insert("\"")
            + pynutil.insert(" pause_length: \"")
            + convert_space(pynutil.insert("PAUSE_LONG phrase_break: true type: PUNCT"))
            + pynutil.insert("\" }")
        )

        graph = medium | loong

        self.fst = graph.optimize()
