# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright 2017 Google Inc.
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

# Adapted from https://github.com/google/TextNormalizationCoveringGrammars
# Russian minimally supervised number grammar.

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    insert_space,
    NEMO_ALPHA,
    delete_space
)
from nemo_text_processing.inverse_text_normalization.de.taggers.cardinal import CardinalFst as ITNCardinalFst
from nemo_text_processing.text_normalization.de.utils import get_abs_path


try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g. 
        "1 001" ->  cardinal { integer: "тысяча один" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        self.graph = ITNCardinalFst().graph_no_exception.invert()
        self.graph = self.graph @ pynini.cdrewrite(delete_space, "[BOS]", "", NEMO_SIGMA)

        self.optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space, 0, 1
        )

        self.cardinal_numbers_with_optional_negative = (
            self.optional_graph_negative
            + pynutil.insert("integer: \"")
            + self.graph
            + pynutil.insert("\"")
        )

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))

        single_digits_graph = pynini.invert(graph_digit | graph_zero)
        self.single_digits_graph = single_digits_graph + pynini.closure(insert_space + single_digits_graph)

        
        serial_graph = self.get_serial_graph()
        final_graph = pynutil.add_weight(self.cardinal_numbers_with_optional_negative, -0.1)
        final_graph |= (
            pynutil.insert("integer: \"")
            + pynutil.add_weight(self.single_digits_graph | serial_graph, 10)
            + pynutil.insert("\"")
        )
        self.final_graph = final_graph
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

    def get_serial_graph(self):
        """
        Finite state transducer for classifying serial.
            The serial is a combination of digits, letters and dashes, e.g.:
            c325-b -> tokens { cardinal { integer: "c three two five b" } }
        """
        alpha = NEMO_ALPHA

        if self.deterministic:
            num_graph = self.single_digits_graph
        else:
            num_graph = self.graph | self.single_digits_graph

        delimiter = insert_space | pynini.cross("-", " ") | pynini.cross("/", " ")
        letter_num = pynini.closure(alpha + delimiter, 1) + num_graph
        num_letter = pynini.closure(num_graph + delimiter, 1) + alpha
        num_delimiter_num = pynini.closure(num_graph + delimiter, 1) + num_graph
        next_alpha_or_num = pynini.closure(delimiter + (alpha | num_graph))
        serial_graph = (letter_num | num_letter | num_delimiter_num) + next_alpha_or_num
        if not self.deterministic:
            serial_graph += pynini.closure(pynini.accep("s"), 0, 1)

        return serial_graph.optimize()
