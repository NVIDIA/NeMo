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

from nemo_text_processing.inverse_text_normalization.de.graph_utils import NEMO_DIGIT, NEMO_SPACE, GraphFst
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

        e.g. minus drei und zwanzig -> cardinal { negative: "-" integer: "23" } }
        e.g. minusdreiundzwanzig -> cardinal { integer: "23" } }
        e.g. dreizehn -> cardinal { integer: "13" } }
        e.g. hundert -> cardinal { integer: "100" } }
        e.g. einhundert -> cardinal { integer: "100" } }
        e.g. tausend -> cardinal { integer: "1000" } }
        e.g. eintausend -> cardinal { integer: "1000" } }
        e.g. tausendundzwanzig -> cardinal { integer: "1020" } }
        e.g. hundertundzwanzig -> cardinal { integer: "120" } }
    
    """

    def __init__(self, tn_cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        graph = tn_cardinal.graph.invert().optimize()
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            tn_cardinal.graph_hundred_component_at_least_one_none_zero_digit.invert().optimize()
        )

        self.graph_ties = tn_cardinal.two_digit_non_zero.invert().optimize()

        self.graph_no_exception = graph
        self.digit = tn_cardinal.digit.invert().optimize()
        graph_exception = pynini.project(self.digit, 'input')
        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("minus ", "\"-\" "), 0, 1)

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
