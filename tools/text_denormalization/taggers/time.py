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

from typing import Union

import inflect
import pynini
from denormalization.data_loader_utils import get_abs_path
from denormalization.graph_utils import (
    NEMO_NON_BREAKING_SPACE,
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
)
from denormalization.taggers.cardinal import CardinalFst
from denormalization.utils import num_to_word
from pynini.lib import pynutil


class TimeFst(GraphFst):
    def __init__(self):
        super().__init__(name="time", kind="classify")
        # hours, minutes, seconds, suffix, zone, style, speak_period

        suffix_graph = pynini.string_file(get_abs_path("data/time.tsv"))

        cardinal = CardinalFst().graph_no_exception

        add_space = pynutil.insert(" ")

        labels_hour = [num_to_word(x) for x in range(1, 13)]
        labels_minute_single = [num_to_word(x) for x in range(1, 10)]
        labels_minute_double = [num_to_word(x) for x in range(10, 60)]

        graph_hour = pynini.union(*labels_hour) @ cardinal

        graph_minute_single = pynini.union(*labels_minute_single) @ cardinal

        graph_minute_double = pynini.union(*labels_minute_double) @ cardinal

        graph_minute_verbose = pynini.cross("half", "30") | pynini.cross("quarter", "15")

        oclock = pynini.cross(pynini.union("o' clock", "o clock", "o'clock"), "")

        # five o' clock
        # two o eight, two thiry five (am/pm)
        # two pm/am
        graph1 = (
            pynutil.insert("hours: \"")
            + graph_hour
            + pynutil.insert("\"")
            + delete_space
            + pynini.union(
                oclock,
                pynini.cross("o", "")
                + delete_extra_space
                + pynutil.insert("minutes: \"")
                + graph_minute_single
                + pynutil.insert("\""),
                delete_extra_space + pynutil.insert("minutes: \"") + graph_minute_double + pynutil.insert("\""),
            )
            + (delete_extra_space + pynutil.insert("suffix: \"") + convert_space(suffix_graph) + pynutil.insert("\""))
            ** (0, 1)
        )
        # 10 to three, 10 past four, quarter past four, half past four
        graph2 = (
            pynutil.insert("minutes: \"")
            + pynini.union(graph_minute_single, graph_minute_double, graph_minute_verbose)
            + pynutil.insert("\"")
            + delete_space
            + (pynini.cross("to", "") | pynini.cross("past", ""))
            + delete_extra_space
            + pynutil.insert("hours: \"")
            + graph_hour
            + pynutil.insert("\"")
            + (delete_extra_space + pynutil.insert("suffix: \"") + convert_space(suffix_graph) + pynutil.insert("\""))
            ** (0, 1)
        )
        final_graph = pynini.union(graph1, graph2).optimize()

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
