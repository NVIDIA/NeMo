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


from nemo_text_processing.text_normalization.de.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    GraphFst,
    convert_space,
    delete_space,
    insert_space,
)

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        "02:15" -> time { hours: "два часа пятнадцать минут" }
    
    Args:
        number_names: number_names for cardinal and ordinal numbers
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        final_suffix = pynutil.delete("Uhr") | pynutil.delete("uhr")
        time_zone_graph = pynini.string_file(get_abs_path("data/time/time_zone.tsv"))

        # only used for < 1000 thousand -> 0 weight
        cardinal = cardinal.graph

        labels_hour = [str(x) for x in range(0, 24)]
        labels_minute_single = [str(x) for x in range(1, 10)]
        labels_minute_double = [str(x) for x in range(10, 60)]

        delete_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (
            pynini.closure(pynutil.delete("0"), 0, 1) + NEMO_DIGIT
        )

        graph_hour = delete_leading_zero_to_double_digit @ pynini.union(*labels_hour) @ cardinal

        graph_minute_single = pynini.union(*labels_minute_single) @ cardinal
        graph_minute_double = pynini.union(*labels_minute_double) @ cardinal

        final_graph_hour = pynutil.insert("hours: \"") + graph_hour + pynutil.insert("\"")
        final_graph_minute = (
            pynutil.insert("minutes: \"")
            + (pynutil.delete("0") + insert_space + graph_minute_single | graph_minute_double)
            + pynutil.insert("\"")
        )
        final_graph_second = (
            pynutil.insert("seconds: \"")
            + (pynutil.delete("0") + insert_space + graph_minute_single | graph_minute_double)
            + pynutil.insert("\"")
        )
        final_suffix_optional = pynini.closure(delete_space + final_suffix, 0, 1)
        final_time_zone_optional = pynini.closure(
            delete_space
            + insert_space
            + pynutil.insert("zone: \"")
            + convert_space(time_zone_graph)
            + pynutil.insert("\""),
            0,
            1,
        )

        # 2:30 Uhr, 02:30, 2:00
        graph_hm = (
            final_graph_hour
            + pynutil.delete(":")
            + (pynutil.delete("00") | insert_space + final_graph_minute)
            + final_suffix_optional
            + final_time_zone_optional
        )

        # 10:30:05 Uhr,
        graph_hms = (
            final_graph_hour
            + pynutil.delete(":")
            + (pynini.cross("00", " minutes: \"null\"") | insert_space + final_graph_minute)
            + pynutil.delete(":")
            + (pynini.cross("00", " seconds: \"null\"") | insert_space + final_graph_second)
            + final_suffix_optional
            + final_time_zone_optional
        )

        # 2.xx Uhr
        graph_hm2 = (
            final_graph_hour
            + pynutil.delete(".")
            + (pynutil.delete("00") | insert_space + final_graph_minute)
            + delete_space
            + insert_space
            + final_suffix
            + final_time_zone_optional
        )
        # 2 Uhr est
        graph_h = final_graph_hour + delete_space + insert_space + final_suffix + final_time_zone_optional
        final_graph = (graph_hm | graph_h | graph_hm2 | graph_hms).optimize()
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
