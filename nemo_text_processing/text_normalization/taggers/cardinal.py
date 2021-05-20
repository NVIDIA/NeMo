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


from nemo_text_processing.text_normalization.data_loader_utils import get_abs_path
from nemo_text_processing.text_normalization.graph_utils import NEMO_DIGIT, GraphFst
from nemo_text_processing.text_normalization.taggers.date import get_hundreds_graph

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g. 
        -23 -> cardinal { negative: "true"  integer: "twenty three" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        graph = pynini.Far(get_abs_path("data/numbers/cardinal_number_name.far")).get_fst()
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            pynini.closure(NEMO_DIGIT, 2, 3) | pynini.difference(NEMO_DIGIT, pynini.accep("0"))
        ) @ graph
        self.graph = (
            pynini.closure(NEMO_DIGIT, 1, 3)
            + pynini.closure(pynini.closure(pynutil.delete(","), 0, 1) + NEMO_DIGIT + NEMO_DIGIT + NEMO_DIGIT)
        ) @ graph

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        single_digits_graph = pynini.invert(graph_digit | graph_zero) | pynini.cross("0", "oh")
        self.single_digits_graph = single_digits_graph + pynini.closure(pynutil.insert(" ") + single_digits_graph)

        if not deterministic:
            single_digits_graph_with_commas = (
                pynini.closure(self.single_digits_graph, 1, 3)
                + pynutil.insert(" ")
                + pynini.closure(
                    pynutil.delete(",")
                    + pynutil.insert(" ")
                    + single_digits_graph
                    + pynutil.insert(" ")
                    + single_digits_graph
                    + pynutil.insert(" ")
                    + single_digits_graph,
                    1,
                )
            )

            self.graph = self.graph | self.single_digits_graph | get_hundreds_graph() | single_digits_graph_with_commas

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)
        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
