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


from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SPACE, GraphFst
from nemo_text_processing.text_normalization.ru.utils import get_abs_path

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time, e.g.
        "02:15" ->
    
    Args:
        number_names: Number_names graph
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, number_names: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        """
        TODO
        handle seconds: 11:35:00
        """
        increment_hour = pynini.string_file(get_abs_path("data/time/increment_hour.tsv"))
        convert_hour = pynini.string_file(get_abs_path("data/time/time_convert.tsv"))
        hour_names = pynini.string_file(get_abs_path("data/time/hour_names.tsv"))

        number = pynini.closure(pynini.cross("0", ""), 0, 1) + number_names.nominative_up_to_thousand_names
        hour_options = pynini.project(increment_hour, "input")
        hour_options = hour_options | pynini.project(convert_hour, "output")

        hour_exeption_ends_with_one = pynini.union(*["01", "21"])
        hour_exeption_ends_rest = pynini.union(*["02", "03", "04", "22", "23"])
        hour_other = (
            pynini.difference(hour_options, pynini.union(hour_exeption_ends_with_one, hour_exeption_ends_rest))
        ).optimize()

        hour = hour_exeption_ends_with_one @ number + pynutil.insert(" час")
        hour |= hour_exeption_ends_rest @ number + pynutil.insert(" часа")
        hour |= hour_other @ number + pynutil.insert(" часов")

        optional_and = pynini.closure(pynutil.insert("и "), 0, 1)
        digits = pynini.union(*[str(x) for x in range(10)])
        mins_start = pynini.union(*"012345")
        mins_options = mins_start + digits
        mins_exception_ends_with_one = mins_start + pynini.accep("1")
        mins_exception_ends_rest = pynini.difference(
            mins_start + pynini.union(*"234"), pynini.union(*["12", "13", "14"])
        )
        mins_other = pynini.difference(
            mins_options, pynini.union(mins_exception_ends_with_one, mins_exception_ends_rest)
        )

        minutes = mins_exception_ends_with_one @ number + pynutil.insert(" минута")
        minutes |= mins_exception_ends_rest @ number + pynutil.insert(" минуты")
        minutes |= mins_other @ number + pynutil.insert(" минут")

        # 17:15 -> "пятнадцать минут шестого"
        self.m_next_h = (
            pynutil.insert("hours: \"")
            + pynini.compose(hour_options, increment_hour)
            + pynutil.insert("\"")
            + pynini.cross(":", " ")
            + pynutil.insert("minutes: \"")
            + minutes
            + pynutil.insert("\"")
        )
        self.hm = (
            pynutil.insert("hours: \"")
            + hour.optimize()
            + pynutil.insert("\"")
            + (pynini.cross(":", " ") + pynutil.insert("minutes: \"") + optional_and + minutes.optimize())
            + pynutil.insert("\"")
            + pynutil.insert(" preserve_order: true")
        )
        self.h = pynutil.insert("hours: \"") + hour + pynutil.insert("\"") + pynutil.delete(":00")

        self.fst = self.m_next_h | self.hm | self.h
        self.fst = self.add_tokens(self.fst)
        self.fst = self.fst.optimize()

        # from pynini.lib.rewrite import top_rewrites
        # import pdb; pdb.set_trace()
        # print(top_rewrites("17:15", self.mins_next_hour, 5))

        # from pynini.lib.rewrite import top_rewrites
        # import pdb; pdb.set_trace()
        # print(top_rewrites("13:05", self.fst, 5))
        # print()

        # suffix_graph = pynini.string_file(get_abs_path("data/time_suffix.tsv"))
        # time_zone_graph = pynini.string_file(get_abs_path("data/time_zone.tsv"))
        #
        # # only used for < 1000 thousand -> 0 weight
        # cardinal = cardinal.graph
        #
        # labels_hour = [str(x) for x in range(0, 24)]
        # labels_minute_single = [str(x) for x in range(1, 10)]
        # labels_minute_double = [str(x) for x in range(10, 60)]
        #
        # delete_leading_zero_to_double_digit = (NEMO_DIGIT + NEMO_DIGIT) | (
        #     pynini.closure(pynutil.delete("0"), 0, 1) + NEMO_DIGIT
        # )
        #
        # graph_hour = delete_leading_zero_to_double_digit @ pynini.union(*labels_hour) @ cardinal
        #
        # graph_minute_single = pynini.union(*labels_minute_single) @ cardinal
        # graph_minute_double = pynini.union(*labels_minute_double) @ cardinal
        #
        # final_graph_hour = pynutil.insert("hours: \"") + graph_hour + pynutil.insert("\"")
        # final_graph_minute = (
        #     pynutil.insert("minutes: \"")
        #     + (pynini.cross("0", "o") + insert_space + graph_minute_single | graph_minute_double)
        #     + pynutil.insert("\"")
        # )
        # final_suffix = pynutil.insert("suffix: \"") + convert_space(suffix_graph) + pynutil.insert("\"")
        # final_suffix_optional = pynini.closure(delete_space + insert_space + final_suffix, 0, 1)
        # final_time_zone_optional = pynini.closure(
        #     delete_space
        #     + insert_space
        #     + pynutil.insert("zone: \"")
        #     + convert_space(time_zone_graph)
        #     + pynutil.insert("\""),
        #     0,
        #     1,
        # )
        #
        # 2:30 pm, 02:30, 2:00

        #
        # # 2.xx pm/am
        # graph_hm2 = (
        #     final_graph_hour
        #     + pynutil.delete(".")
        #     + (pynutil.delete("00") | insert_space + final_graph_minute)
        #     + delete_space
        #     + insert_space
        #     + final_suffix
        #     + final_time_zone_optional
        # )
        # # 2 pm est
        # graph_h = final_graph_hour + delete_space + insert_space + final_suffix + final_time_zone_optional
        # final_graph = (graph_hm | graph_h | graph_hm2).optimize()
        #
