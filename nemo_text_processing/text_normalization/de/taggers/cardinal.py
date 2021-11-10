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

from nemo_text_processing.inverse_text_normalization.de.taggers.cardinal import CardinalFst as ITNCardinalFst
from nemo_text_processing.text_normalization.de.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from collections import defaultdict

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False

AND="und"
def get_ties_digit(digit_path: str, tie_path: str):
    """
    getting all inverse normalizations for numbers between 21 - 100
    Args:
        digit_path: file to digit tsv
        tie_path: file to tie tsv, e.g. 20, 30, etc.
    """

    digits = defaultdict(list)
    ties = defaultdict(list)
    for k, v in load_labels(digit_path):
        digits[v].append(k)

    for k, v in load_labels(tie_path):
        ties[v].append(k)

    d = []
    for i in range(21, 100):
        s = str(i)
        if s[1] == "0":
            continue

        for di in digits[s[1]]:
            for ti in ties[s[0]]:
                word = di + f" {AND} " + ti
                d.append((word, s))

    return pynini.string_map(d)


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g. 
        "1 001" ->  cardinal { integer: "тысяча один" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)


    need self.graph_hundred_component_at_least_one_none_zero_digit
    self.graph
    self.cardinal_numbers_with_optional_negative
    self.single_digits_graph

    """

    def __init__(self, deterministic: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv")).invert()
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv")).invert()

        separator = "."


        def tens_no_zero():
            return ( pynutil.delete("0") + graph_digit | get_ties_digit(get_abs_path("data/numbers/digit.tsv"), get_abs_path("data/numbers/ties.tsv")).invert()| graph_teen)

        def hundred_non_zero():
            return (graph_digit + insert_space + pynutil.insert("hundert") + (pynini.closure( insert_space + pynutil.insert(AND), 0, 1) + insert_space + tens_no_zero() | pynutil.delete("00"))  | pynutil.delete("0") + tens_no_zero())

        def thousand():
            return (hundred_non_zero() + insert_space +  pynutil.insert("tausend") | pynutil.delete("000"))  +  (insert_space + hundred_non_zero() | pynutil.delete("000"))

        graph_million = (
            pynini.union(
                hundred_non_zero() + insert_space + pynutil.insert("million") + pynini.closure(pynutil.insert("en"), 0, 1),
                pynutil.delete("000")
            )
        )

        graph_billion = (
            pynini.union(
                hundred_non_zero() + insert_space + pynutil.insert("milliarde") + pynini.closure(pynutil.insert("n"), 0, 1),
                pynutil.delete("000")
            )
        )

        graph_trillion  = (
            pynini.union(
                hundred_non_zero() + insert_space + pynutil.insert("billion") + pynini.closure(pynutil.insert("en"), 0, 1),
                pynutil.delete("000")
            )
        )

        
        graph_quadrillion  = (
            pynini.union(
                hundred_non_zero() + insert_space + pynutil.insert("billiarde") + pynini.closure(pynutil.insert("n"), 0, 1),
                pynutil.delete("000")
            )
        )
        
        graph_quintillion   = (
            pynini.union(
                hundred_non_zero() + insert_space + pynutil.insert("trillion") + pynini.closure(pynutil.insert("en"), 0, 1),
                pynutil.delete("000")
            )
        )

        graph_sextillion    = (
            pynini.union(
                hundred_non_zero() + insert_space + pynutil.insert("trilliarde") + pynini.closure(pynutil.insert("n"), 0, 1),
                pynutil.delete("000")
            )
        )
        graph = pynini.union(
            graph_sextillion
            + insert_space
            + graph_quintillion
            + insert_space
            + graph_quadrillion
            + insert_space
            + graph_trillion
            + insert_space
            + graph_billion
            + insert_space
            + graph_million
            + insert_space
            + thousand()
        )

        self.graph = pynini.difference(pynini.closure(NEMO_DIGIT, 1), "0") @ pynini.cdrewrite(pynini.closure(pynutil.insert("0")), "[BOS]", "", NEMO_SIGMA) @ NEMO_DIGIT ** 24 @ graph @ pynini.cdrewrite(delete_space, "[BOS]", "", NEMO_SIGMA)
        self.graph |= graph_zero

        self.graph = pynini.cdrewrite(pynutil.delete(separator), "", "", NEMO_SIGMA) @ self.graph
        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
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
