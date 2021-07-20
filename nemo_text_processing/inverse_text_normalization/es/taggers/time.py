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


from nemo_text_processing.inverse_text_normalization.es.graph_utils import (
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.inverse_text_normalization.es.taggers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.es.utils import get_abs_path, num_to_word

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time
        e.g. twelve thirty -> time { hours: "12" minutes: "30" }
        e.g. twelve past one -> time { minutes: "12" hours: "1" }
        e.g. two o clock a m -> time { hours: "2" suffix: "a.m." }
        e.g. quarter to two -> time { hours: "1" minutes: "45" }
        e.g. quarter past two -> time { hours: "2" minutes: "15" }
        e.g. half past two -> time { hours: "2" minutes: "30" }
    """

    def __init__(self):
        super().__init__(name="time", kind="classify")
        # hours, minutes, seconds, suffix, zone, style, speak_period

        suffix_graph = pynini.string_file(get_abs_path("data/time/time_suffix.tsv"))
        time_to_graph = pynini.string_file(get_abs_path("data/time/time_to.tsv"))

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
        graph_twenties = pynini.string_file(get_abs_path("data/numbers/twenties.tsv"))

        single_digits = graph_digit | pynini.cross("0", "cero")

        double_digits = pynini.union(
            graph_twenties,
            graph_teen,
            (graph_ties + pynutil.insert("0")),
            (graph_ties + pynutil.delete(" y ") + graph_digit),
        )

        graph_0_to_100 = pynini.union(single_digits, double_digits)

        # note that making graph_hours starting from 2 hours
        # "1 o'clock" will be treated differently because it
        # is singular
        digits_2_to_24 = [str(digits) for digits in range(2, 24)]
        digits_1_to_60 = [str(digits) for digits in range(1, 60)]

        graph_1oclock = pynini.cross("la una", "la 1")
        graph_hour = pynini.cross("las ", "las ") + graph_0_to_100 @ pynini.union(*digits_2_to_24)
        graph_minute = graph_0_to_100 @ pynini.union(*digits_1_to_60)
        graph_minute_verbose = pynini.cross("media", "30") | pynini.cross("cuarto", "15")

        final_graph_hour = pynutil.insert("hours: \"") + (graph_1oclock | graph_hour) + pynutil.insert("\"")

        final_graph_minute = (
            pynutil.insert("minutes: \"")
            + pynini.closure((pynutil.delete("y") | pynutil.delete("con")) + delete_space, 0, 1)
            + (graph_minute | graph_minute_verbose)
            + pynutil.insert("\"")
        )

        final_suffix = pynutil.insert("suffix: \"") + convert_space(suffix_graph) + pynutil.insert("\"")
        final_suffix_optional = pynini.closure(delete_space + insert_space + final_suffix, 0, 1)

        # las nueve y veinticinco
        graph_hm = final_graph_hour + delete_extra_space + final_graph_minute

        # un cuarto para las cinco
        graph_mh = (
            pynutil.insert("minutes: \"")
            + pynini.union(pynini.cross("un cuarto para", "45"), pynini.cross("cuarto para", "45"),)
            + pynutil.insert("\"")
            + delete_extra_space
            + pynutil.insert("hours: \"")
            + time_to_graph
            + pynutil.insert("\"")
        )

        # las diez menos diez
        graph_time_to = (
            pynutil.insert("hours: \"")
            + time_to_graph
            + pynutil.insert("\"")
            + delete_extra_space
            + pynutil.insert("minutes: \"")
            + delete_space
            + pynutil.delete("menos")
            + delete_space
            + pynini.union(
                pynini.cross("cinco", "55"),
                pynini.cross("diez", "50"),
                pynini.cross("cuarto", "45"),
                pynini.cross("veinte", "40"),
                pynini.cross("veinticinco", "30"),
            )
            + pynutil.insert("\"")
        )
        final_graph = ((graph_hm | graph_mh | graph_time_to) + final_suffix_optional).optimize()

        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
