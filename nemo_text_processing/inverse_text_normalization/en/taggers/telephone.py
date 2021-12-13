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

from nemo_text_processing.inverse_text_normalization.en.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, GraphFst, insert_space

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g. 
        one two three one two three five six seven eight -> { number_part: "123-123-5678" }
    """

    def __init__(self):
        super().__init__(name="telephone", kind="classify")
        # country code, number_part, extension
        digit_to_str = pynini.invert(
            pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        ).optimize() | pynini.cross("0", pynini.union("o", "oh", "zero"))

        double_digit = pynini.union(
            *[
                pynini.cross(
                    pynini.project(str(i) @ digit_to_str, "output")
                    + pynini.accep(" ")
                    + pynini.project(str(i) @ digit_to_str, "output"),
                    pynutil.insert("double ") + pynini.project(str(i) @ digit_to_str, "output"),
                )
                for i in range(10)
            ]
        )
        double_digit.invert()
        number_part = (
            pynini.closure(digit_to_str + insert_space, 2, 2)
            + digit_to_str
            + pynutil.delete("-")
            + insert_space
            + pynini.closure(digit_to_str + insert_space, 2, 2)
            + digit_to_str
            + pynutil.delete("-")
            + insert_space
            + pynini.closure(digit_to_str + insert_space, 3, 3)
            + digit_to_str
        )
        number_part = (
            pynutil.insert("number_part: \"")
            + pynini.cdrewrite(double_digit, "", "", NEMO_SIGMA) @ pynini.invert(number_part)
            + pynutil.insert("\"")
        )

        graph = number_part
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
