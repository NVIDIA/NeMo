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


from nemo_text_processing.text_normalization.en.graph_utils import NEMO_ALPHA, NEMO_DIGIT, GraphFst, insert_space
from nemo_text_processing.text_normalization.en.taggers.date import get_hundreds_graph
from nemo_text_processing.text_normalization.en.utils import get_abs_path, load_labels

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

        single_digits_graph = pynini.invert(graph_digit | graph_zero)
        self.single_digits_graph = single_digits_graph + pynini.closure(pynutil.insert(" ") + single_digits_graph)

        if not deterministic:
            single_digits_graph = pynutil.add_weight(
                pynini.invert(graph_digit | graph_zero), 1.2
            ) | pynutil.add_weight(pynini.cross("0", "oh"), 1.1)
            self.single_digits_graph = single_digits_graph + pynini.closure(pynutil.insert(" ") + single_digits_graph)

            single_digits_graph_with_commas = pynini.closure(
                self.single_digits_graph + pynutil.insert(" "), 1, 3
            ) + pynini.closure(
                pynutil.delete(",")
                + single_digits_graph
                + pynutil.insert(" ")
                + single_digits_graph
                + pynutil.insert(" ")
                + single_digits_graph,
                1,
            )
            self.graph |= self.single_digits_graph | get_hundreds_graph() | single_digits_graph_with_commas
            self.range_graph = (
                pynini.closure(pynutil.insert("from "), 0, 1)
                + self.graph
                + (pynini.cross("-", " to ") | pynini.cross("-", " "))
                + self.graph
            )

            self.range_graph |= self.graph + (pynini.cross("x", " by ") | pynini.cross(" x ", " by ")) + self.graph
            self.range_graph = self.range_graph.optimize()

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        long_numbers = pynini.compose(NEMO_DIGIT ** (5, ...), self.single_digits_graph).optimize()
        final_graph = (
            pynutil.add_weight(self.graph, 1.2)
            | pynutil.add_weight(self.get_serial_graph(), 1.2)
            | pynutil.add_weight(long_numbers, 1.1)
        )

        if not deterministic:
            final_graph |= self.range_graph

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + final_graph + pynutil.insert("\"")
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
            num_graph = self.graph
            letter_pronunciation = pynini.string_map(load_labels(get_abs_path("data/letter_pronunciation.tsv")))
            alpha |= letter_pronunciation

        delimiter = insert_space | pynini.cross("-", " ") | pynini.cross("/", " ")
        letter_num = pynini.closure(alpha + delimiter, 1) + num_graph
        num_letter = pynini.closure(num_graph + delimiter, 1) + alpha
        next_alpha_or_num = pynini.closure(delimiter + (alpha | num_graph))
        serial_graph = (letter_num | num_letter) + next_alpha_or_num

        if not self.deterministic:
            serial_graph += pynini.closure(pynini.accep("s") | pynini.cross("s", "es"), 0, 1)
        return pynutil.add_weight(serial_graph, 10)
