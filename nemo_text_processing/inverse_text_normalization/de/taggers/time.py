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


from nemo_text_processing.inverse_text_normalization.de.graph_utils import (
    GraphFst,
    convert_space,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time
        e.g. acht uhr -> time { hours: "8" minutes: "00" }
        e.g. dreizehn uhr -> time { hours: "13" minutes: "00" }
        e.g. dreizehn uhr zehn -> time { hours: "13" minutes: "10" }
        e.g. acht uhr abends -> time { hours: "8" minutes: "00" suffix: "abends"}
        e.g. acht uhr nachmittags -> time { hours: "8" minutes: "00" suffix: "nachmittags"}
        e.g. viertel vor zwölf -> time { minutes: "45" hours: "11" }
        e.g. viertel nach zwölf -> time { minutes: "15" hours: "12" }
        e.g. halb zwölf -> time { minutes: "30" hours: "11" }
        e.g. viertel zwölf -> time { minutes: "15" hours: "11" }
        e.g. drei minuten vor zwölf -> time { minutes: "57" hours: "11" }
        e.g. drei vor zwölf -> time { minutes: "57" hours: "11" }
        e.g. drei minuten nach zwölf -> time { minutes: "03" hours: "12" }
        e.g. drei viertel zwölf -> time { minutes: "45" hours: "11" }
    """

    def __init__(self):
        super().__init__(name="time", kind="classify")
        # hours, minutes, seconds, suffix, zone, style, speak_period

        time_zone = pynini.invert(pynini.string_file(get_abs_path("data/time/time_zone.tsv")))
        hour_to = pynini.string_file(get_abs_path("data/time/hour_to.tsv"))
        minute_to = pynini.string_file(get_abs_path("data/time/minute_to.tsv"))
        hour = pynini.string_file(get_abs_path("data/time/hour.tsv"))
        minute = pynini.string_file(get_abs_path("data/time/minute.tsv"))
        half = pynini.cross("halb", "30")
        quarters = (
            pynini.cross("viertel", "15") | pynini.cross("drei viertel", "45") | pynini.cross("dreiviertel", "45")
        )
        oclock = pynutil.delete("uhr")

        final_graph_hour = pynutil.insert("hours: \"") + hour + pynutil.insert("\"")
        # "[..] uhr (zwanzig)"
        final_graph_minute = (
            oclock
            + pynutil.insert("minutes: \"")
            + (pynutil.insert("00") | delete_space + minute)
            + pynutil.insert("\"")
        )
        final_time_zone_optional = pynini.closure(
            delete_space + insert_space + pynutil.insert("zone: \"") + convert_space(time_zone) + pynutil.insert("\""),
            0,
            1,
        )

        # vier uhr
        # vier uhr zehn
        # vierzehn uhr zehn
        graph_hm = final_graph_hour + delete_extra_space + final_graph_minute

        # zehn nach vier, vierzehn nach vier, viertel nach vier
        graph_m_nach_h = (
            pynutil.insert("minutes: \"")
            + pynini.union(minute + pynini.closure(delete_space + pynutil.delete("minuten"), 0, 1), quarters)
            + pynutil.insert("\"")
            + delete_space
            + pynutil.delete("nach")
            + delete_extra_space
            + pynutil.insert("hours: \"")
            + hour
            + pynutil.insert("\"")
        )

        # 10 vor vier,  viertel vor vier
        graph_m_vor_h = (
            pynutil.insert("minutes: \"")
            + pynini.union(minute + pynini.closure(delete_space + pynutil.delete("minuten"), 0, 1), quarters)
            @ minute_to
            + pynutil.insert("\"")
            + delete_space
            + pynutil.delete("vor")
            + delete_extra_space
            + pynutil.insert("hours: \"")
            + hour @ hour_to
            + pynutil.insert("\"")
        )

        # viertel zehn,  drei viertel vier, halb zehn
        graph_mh = (
            pynutil.insert("minutes: \"")
            + pynini.union(half, quarters)
            + pynutil.insert("\"")
            + delete_extra_space
            + pynutil.insert("hours: \"")
            + hour @ hour_to
            + pynutil.insert("\"")
        )

        # suffix
        optional_graph_suffix = pynini.closure(
            delete_extra_space
            + pynutil.insert("suffix: \"")
            + pynini.union('abends', 'nachmittags')
            + pynutil.insert("\""),
            0,
            1,
        )

        final_graph = (
            (graph_hm | graph_mh | graph_m_vor_h | graph_m_nach_h) + optional_graph_suffix + final_time_zone_optional
        ).optimize()

        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
