# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


from collections import defaultdict

from nemo_text_processing.inverse_text_normalization.de.graph_utils import (
    NEMO_DIGIT,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path, load_labels

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


AND = "und"


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
                word = di + f"{AND}" + ti
                d.append((word, s))

    return pynini.string_map(d)


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals. Numbers below ten are not converted. 
    Allows both compound numeral strings or separated by whitespace.
    "und" (en: "and") can be inserted between "hundert" and following number or "tausend" and following single or double digit number.

        e.g. minus drei und zwanzig -> cardinal { integer: "23" negative: "-" } }
        e.g. minusdreiundzwanzig -> cardinal { integer: "23" } }
        e.g. dreizehn -> cardinal { integer: "13" } }
        e.g. hundert -> cardinal { integer: "100" } }
        e.g. einhundert -> cardinal { integer: "100" } }
        e.g. tausend -> cardinal { integer: "1000" } }
        e.g. eintausend -> cardinal { integer: "1000" } }
        e.g. tausendundzwanzig -> cardinal { integer: "1020" } }
        e.g. hundertundzwanzig -> cardinal { integer: "120" } }
    
    """

    def __init__(self):
        super().__init__(name="cardinal", kind="classify")
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))

        file_hundred = pynini.string_file(get_abs_path("data/numbers/hundred.tsv"))

        graph_hundred = pynutil.delete(file_hundred)

        graph_ties_digit = get_ties_digit(
            get_abs_path("data/numbers/digit.tsv"), get_abs_path("data/numbers/ties.tsv")
        )
        graph_ties = graph_ties_digit | (graph_ties + pynutil.insert("0"))
        self.graph_ties = graph_ties.optimize()

        graph_hundred_component = pynini.union(
            pynini.union(graph_digit + delete_space, pynutil.insert('1')) + graph_hundred, pynutil.insert("0")
        )
        graph_hundred_component += delete_space
        graph_hundred_component += pynini.union(
            pynutil.insert("00"),
            pynini.closure(pynutil.delete(AND) + delete_space, 0, 1)
            + pynini.union(
                graph_teen, graph_ties, pynutil.insert("0") + graph_digit,  #  fourteen  # twenty, twenty four,
            ),
        )

        graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component
            @ (pynini.closure(NEMO_DIGIT) + (NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT)).optimize()
        )
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            graph_hundred_component_at_least_one_none_zero_digit
        )

        graph_thousands = pynini.union(
            pynini.union(graph_hundred_component_at_least_one_none_zero_digit + delete_space, pynutil.insert('1'))
            + pynutil.delete("tausend"),
            pynutil.insert("000", weight=0.1),
        )

        graph_million = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("million")
            + pynini.closure(pynutil.delete("en"), 0, 1),
            pynutil.insert("000", weight=0.1),
        )
        graph_billion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("milliarde")
            + pynini.closure(pynutil.delete("n"), 0, 1),
            pynutil.insert("000", weight=0.1),
        )
        graph_trillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("billion")
            + pynini.closure(pynutil.delete("en"), 0, 1),
            pynutil.insert("000", weight=0.1),
        )
        graph_quadrillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("billiarde")
            + pynini.closure(pynutil.delete("n"), 0, 1),
            pynutil.insert("000", weight=0.1),
        )
        graph_quintillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("trillion")
            + pynini.closure(pynutil.delete("en"), 0, 1),
            pynutil.insert("000", weight=0.1),
        )
        graph_sextillion = pynini.union(
            graph_hundred_component_at_least_one_none_zero_digit
            + delete_space
            + pynutil.delete("trilliarde")
            + pynini.closure(pynutil.delete("n"), 0, 1),
            pynutil.insert("000", weight=0.1),
        )

        graph = pynini.union(
            graph_sextillion
            + delete_space
            + graph_quintillion
            + delete_space
            + graph_quadrillion
            + delete_space
            + graph_trillion
            + delete_space
            + graph_billion
            + delete_space
            + graph_million
            + delete_space
            + graph_thousands
            + delete_space
            + graph_hundred_component,
            graph_zero,
        )

        graph = (
            graph
            @ pynini.union(
                pynutil.delete(pynini.closure("0")) + pynini.difference(NEMO_DIGIT, "0") + pynini.closure(NEMO_DIGIT),
                "0",
            ).optimize()
        )

        graph_exception = pynini.project(pynini.union(graph_digit, graph_zero), 'input')

        self.graph_no_exception = graph

        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("minus", "\"-\"") + NEMO_SPACE, 0, 1
        )

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
