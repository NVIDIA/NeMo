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

from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


AND = "und"


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals. Numbers below ten are not converted. 
    Allows both compound numeral strings or separated by whitespace.
    "und" (en: "and") can be inserted between "hundert" and following number or "tausend" and following single or double digit number.

        e.g. minus drei und zwanzig -> cardinal { negative: "-" integer: "23" } }
        e.g. minus dreiundzwanzig -> cardinal { integer: "23" } }
        e.g. dreizehn -> cardinal { integer: "13" } }
        e.g. ein hundert -> cardinal { integer: "100" } }
        e.g. einhundert -> cardinal { integer: "100" } }
        e.g. ein tausend -> cardinal { integer: "1000" } }
        e.g. eintausend -> cardinal { integer: "1000" } }
        e.g. ein tausend zwanzig -> cardinal { integer: "1020" } }
    """

    def __init__(self, tn_cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        # add_space_between_chars = pynini.cdrewrite(pynini.closure(insert_space, 0, 1), NEMO_CHAR, NEMO_CHAR, NEMO_SIGMA)
        optional_delete_space = pynini.closure(NEMO_SIGMA | pynutil.delete(" "))

        graph = (tn_cardinal.graph @ optional_delete_space).invert().optimize()
        self.graph_hundred_component_at_least_one_none_zero_digit = (
            (tn_cardinal.graph_hundred_component_at_least_one_none_zero_digit @ optional_delete_space)
            .invert()
            .optimize()
        )

        self.graph_ties = (tn_cardinal.two_digit_non_zero @ optional_delete_space).invert().optimize()

        self.graph_no_exception = graph
        self.digit = tn_cardinal.digit.invert().optimize()
        graph_exception = pynini.project(self.digit, 'input')
        self.graph = (pynini.project(graph, "input") - graph_exception.arcsort()) @ graph

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("minus ", "\"-\" "), 0, 1)

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
