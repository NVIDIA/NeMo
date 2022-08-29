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


import pynini
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, NEMO_SPACE, GraphFst, delete_space
from pynini.lib import pynutil


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fractions
        e.g. fraction { numerator: "8" denominator: "3" } -> "8/3"
    """

    def __init__(self):
        super().__init__(name="fraction", kind="verbalize")

        optional_negative = pynutil.delete("negative: \"") + pynini.cross("True", "-") + pynutil.delete("\"")
        optional_negative = pynini.closure(optional_negative + delete_space, 0, 1)

        integer_part = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")
        optional_integer_part = pynini.closure(integer_part + NEMO_SPACE, 0, 1)

        numerator = pynutil.delete("numerator: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")

        denominator = pynutil.delete("denominator: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\"")

        graph = (
            optional_negative + optional_integer_part + numerator + delete_space + pynutil.insert("/") + denominator
        )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
