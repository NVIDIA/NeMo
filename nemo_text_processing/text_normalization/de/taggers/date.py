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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    TO_LOWER,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.de.utils import get_abs_path

try:
    import pynini
    from pynini.lib import pynutil
    graph_teen = pynini.invert(pynini.string_file(get_abs_path("data/numbers/teen.tsv"))).optimize()
    graph_digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv"))).optimize()
    ties_graph = pynini.invert(pynini.string_file(get_abs_path("data/numbers/ties.tsv"))).optimize()


    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    graph_teen = None
    graph_digit = None
    ties_graph = None
    PYNINI_AVAILABLE = True

def get_year_graph(cardinal: GraphFst, deterministic: bool = True):
    """
    Returns a four digit transducer which is combination of ties/teen or digits
    (using hundred instead of thousand format), e.g.
    1219 -> twelve nineteen
    3900 -> thirty nine hundred
    """
    graph_two_digit = ((NEMO_DIGIT + NEMO_DIGIT) @ cardinal.graph_hundred_component_at_least_one_none_zero_digit)
    graph_two_digit |= pynini.closure(pynutil.delete("0"), 0, 1) + (NEMO_DIGIT @ cardinal.graph_hundred_component_at_least_one_none_zero_digit)
    hundred = pynutil.insert("hundert")
    graph = (
        graph_two_digit + insert_space + graph_two_digit
        | graph_two_digit + insert_space + pynutil.delete("00") + hundred
        | graph_two_digit + insert_space + hundred + pynini.closure(pynutil.insert(" und "), 0, 1) + graph_two_digit
        | pynutil.add_weight(
            NEMO_DIGIT @ cardinal.graph_hundred_component_at_least_one_none_zero_digit
            + insert_space
            + pynini.cross("0", "tausend")
            + (pynutil.delete("00") | (pynini.closure(pynutil.insert(" und "), 0, 1)  + (((pynutil.delete("0") + NEMO_DIGIT)| (NEMO_DIGIT + NEMO_DIGIT))@ cardinal.graph_hundred_component_at_least_one_none_zero_digit))),
            weight=-0.001,
        )
    )
    return graph


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g. 
        "01.05" -> tokens { date { day: "первое мая" } }

    Args:
        number_names: number_names for cardinal and ordinal numbers
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        # DE format: DD-MM-YYYY or DD-MM-YY
        month_abbr_graph = pynini.string_file(get_abs_path("data/months/abbr_to_name.tsv")).optimize()
        month_abbr_graph = (
            month_abbr_graph | ((TO_LOWER + pynini.closure(NEMO_CHAR)) @ month_abbr_graph)
        ) + pynini.closure(pynutil.delete("."), 0, 1)

        month_graph = pynini.string_file(get_abs_path("data/months/names.tsv")).optimize()
        month_graph |= (TO_LOWER + pynini.closure(NEMO_CHAR)) @ month_graph
        month_graph |= month_abbr_graph

        delete_sep = pynutil.add_weight(pynini.cross(".", " "), 1.09) | pynutil.add_weight(
            pynini.cross(pynini.union("/", "-"), " "), 1.1
        ) | delete_extra_space

        numbers = cardinal.graph_hundred_component_at_least_one_none_zero_digit
        digit_day = ((pynini.union("1", "2", "3") + NEMO_DIGIT) @ numbers)
        digit_day |= pynini.closure(pynutil.delete("0"), 0, 1) + (NEMO_DIGIT @ numbers)
        day = (pynutil.insert("day: \"") + digit_day + pynutil.insert("\"")).optimize()

        digit_month = pynini.compose(pynini.accep("1") + NEMO_DIGIT, numbers)
        digit_month |= pynini.closure(pynutil.delete("0"), 0, 1) + (NEMO_DIGIT @ numbers)

        month_number = pynini.string_file(get_abs_path("data/months/numbers.tsv")).optimize()
        month_number = (
            (
                ((pynutil.add_weight(pynutil.delete("0"), -0.1) | pynini.accep("1")) + NEMO_DIGIT) | NEMO_DIGIT
            ).optimize()
            @ month_number
        ).optimize()

        month_abbr_to_names = pynini.string_file(get_abs_path("data/months/abbr_to_name.tsv")).optimize()
        month_names = pynini.string_file(get_abs_path("data/months/months.tsv")).optimize()
        month_name = (
            month_number | month_abbr_to_names | month_names | digit_month
        ).optimize()
        month = (pynutil.insert("month: \"") + month_name + pynutil.insert("\"")).optimize()
        year = pynini.compose(((NEMO_DIGIT ** 4) | (NEMO_DIGIT ** 2)), numbers).optimize()
        year |= get_year_graph(cardinal=cardinal, deterministic=deterministic)

        year_optional = pynutil.insert("year: \"") + year + pynutil.insert("\"")
        year_optional = pynini.closure(delete_sep + year_optional, 0, 1).optimize()
        year_only = pynutil.insert("year: \"") + year +  pynutil.insert("\"")
        
        graph_dmy = day + delete_sep + month + pynini.closure(pynutil.delete(","), 0, 1) + year_optional
        graph_ymd = (
            year
            + delete_sep
            + month
            + delete_sep
            + day
        )


        final_graph = graph_dmy
        if deterministic:
            final_graph += pynutil.insert(" preserve_order: true")
        else:
            final_graph += pynini.closure(pynutil.insert(" preserve_order: true"), 0, 1)
        final_graph |= graph_ymd | year_only

        self.final_graph = final_graph.optimize()
        self.fst = self.add_tokens(self.final_graph).optimize()
