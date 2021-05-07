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

from nemo_text_processing.inverse_text_normalization.utils import get_abs_path
from nemo_text_processing.text_normalization.graph_utils import GraphFst, delete_space, insert_space

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
        delete_space = pynutil.delete(' ')
        # country code, number_part, extension
        add_separator = pynutil.insert(" ")  # between components
        digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv"))).optimize() | pynini.cross(
            "0", pynini.union("o", "oh", "zero")
        )

        number_part = (
            (
                (pynini.closure(digit + insert_space, 2, 2) + digit + pynutil.delete("-"))
                | (
                    pynutil.delete("(")
                    + pynini.closure(digit + insert_space, 2, 2)
                    + digit
                    + pynutil.delete(")")
                    + pynini.closure(pynutil.delete("-"), 0, 1)
                    + delete_space
                )
            )
            + add_separator
            + pynini.closure(digit + insert_space, 2, 2)
            + digit
            + pynutil.delete("-")
            + add_separator
            + pynini.closure(digit + insert_space, 3, 3)
            + digit
        )
        number_part = pynutil.insert("number_part: \"") + pynini.invert(number_part) + pynutil.insert("\"")

        graph = number_part
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
