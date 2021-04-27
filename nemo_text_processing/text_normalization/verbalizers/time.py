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

from nemo_text_processing.text_normalization.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "twelve" minutes: "thirty" suffix: "a m" zone: "e s t" } -> twelve thirty a m e s t
        time { hours: "twelve" } -> twelve o'clock
    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")
        hour = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        minute = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        suffix = (
            pynutil.delete("suffix:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_suffix = pynini.closure(delete_space + insert_space + suffix, 0, 1)
        zone = (
            pynutil.delete("zone:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        optional_zone = pynini.closure(delete_space + insert_space + zone, 0, 1)
        graph = hour + delete_space + insert_space + minute + optional_suffix + optional_zone
        graph |= hour + insert_space + pynutil.insert("o'clock") + optional_zone
        graph |= hour + delete_space + insert_space + suffix + optional_zone
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
