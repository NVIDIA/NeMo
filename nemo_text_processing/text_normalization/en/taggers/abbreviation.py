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
    NEMO_LOWER,
    NEMO_UPPER,
    TO_LOWER,
    GraphFst,
    insert_space,
)

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class AbbreviationFst(GraphFst):
    """
    Finite state transducer for classifying electronic: as URLs, email addresses, etc.
        e.g. "ABC" -> tokens { abbreviation { value: "A B C" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="abbreviation", kind="classify", deterministic=deterministic)

        graph = NEMO_UPPER + pynini.closure(insert_space + NEMO_UPPER, 1)
        graph |= TO_LOWER + pynini.closure(insert_space + TO_LOWER)
        graph |= NEMO_UPPER + pynini.closure(insert_space + NEMO_LOWER, 1)
        graph |= TO_LOWER + pynini.closure(insert_space + NEMO_LOWER)
        graph |= NEMO_UPPER + pynutil.delete(".") + pynini.closure(insert_space + NEMO_UPPER + pynutil.delete("."))
        graph |= TO_LOWER + pynutil.delete(".") + pynini.closure(insert_space + TO_LOWER + pynutil.delete("."))

        graph = pynutil.insert("value: \"") + graph.optimize() + pynutil.insert("\"")
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
