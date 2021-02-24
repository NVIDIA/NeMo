# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from denormalization.graph_utils import NEMO_DIGIT, NEMO_NOT_QUOTE, GraphFst, delete_extra_space, delete_space
from pynini.lib import pynutil


class TimeFst(GraphFst):
    def __init__(self):
        super().__init__(name="time", kind="verbalize")
        add_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (pynutil.insert("0") + NEMO_DIGIT)
        hour = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete("\"")
        )
        minutes = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_DIGIT, 1)
            + pynutil.delete("\"")
        ) ** (0, 1)
        suffix = (
            delete_extra_space
            + pynutil.delete("suffix:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        ) ** (
            0,
            1,
        )  # optional
        graph = (
            hour @ add_leading_zero_to_double_digit
            + delete_space
            + pynutil.insert(":")
            + ((minutes @ add_leading_zero_to_double_digit) | pynutil.insert("00"))
            + suffix
        )
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
