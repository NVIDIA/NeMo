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
        whitelist: whitelist FST
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, whitelist: 'pynini.FstLike', deterministic: bool = True):
        super().__init__(name="abbreviation", kind="classify", deterministic=deterministic)

        main_graph = NEMO_UPPER + pynini.closure(insert_space + NEMO_UPPER, 1)
        misc_graph = pynutil.add_weight(
            TO_LOWER + pynini.closure(insert_space + pynini.union(TO_LOWER | NEMO_LOWER)), 110
        )
        misc_graph |= pynutil.add_weight(
            pynini.closure(NEMO_UPPER, 2) + pynini.closure(insert_space + NEMO_LOWER, 1), 110
        )
        misc_graph |= (
            NEMO_UPPER + pynutil.delete(".") + pynini.closure(insert_space + NEMO_UPPER + pynutil.delete("."))
        )
        misc_graph |= pynutil.add_weight(
            TO_LOWER + pynutil.delete(".") + pynini.closure(insert_space + TO_LOWER + pynutil.delete(".")), 110
        )

        # set weight of the misc graph to the value higher then word
        graph = pynutil.add_weight(main_graph.optimize(), 10) | pynutil.add_weight(misc_graph.optimize(), 101)

        # exclude words that are included in the whitelist
        graph = pynini.compose(
            pynini.difference(pynini.project(graph, "input"), pynini.project(whitelist.graph, "input")), graph
        )
        graph = pynutil.insert("value: \"") + graph.optimize() + pynutil.insert("\"")
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
