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
from denormalization.graph_utils import NEMO_CHAR, NEMO_NOT_QUOTE, GraphFst, delete_space
from pynini.lib import pynutil


class MeasureFst(GraphFst):
    def __init__(self):
        super().__init__(name="measure", kind="verbalize")
        sign = pynini.closure(pynini.cross("negative: \"true\"", "-"), 0, 1)
        integer = (
            (pynutil.delete("integer:") | pynutil.delete("integer_part:"))
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        point = pynutil.insert(".")
        fractional = (
            pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        unit = (
            pynutil.delete("units:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_CHAR - " ", 1)
            + pynutil.delete("\"")
        )
        # graph = sign + delete_space + integer + delete_space + point + delete_space + fractional
        graph = (
            pynini.union(
                pynutil.delete("decimal {")
                + delete_space
                + pynini.closure(integer + delete_space, 0, 1)
                + point
                + delete_space
                + fractional,
                pynutil.delete("cardinal {") + delete_space + integer,
            )
            + delete_space
            + pynutil.delete("}")
            + delete_space
            + pynutil.insert(" ")
            + unit
        )
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
