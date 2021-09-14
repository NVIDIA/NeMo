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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_SIGMA,
    TO_LOWER,
    GraphFst,
    delete_extra_space,
    insert_space,
)
from nemo_text_processing.text_normalization.en.utils import get_abs_path, load_labels

try:
    import pynini
    from pynini.lib import pynutil

    graph_teen = pynini.invert(pynini.string_file(get_abs_path("data/numbers/teen.tsv"))).optimize()
    graph_digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv"))).optimize()
    ties_graph = pynini.invert(pynini.string_file(get_abs_path("data/numbers/ties.tsv"))).optimize()

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    # Add placeholders for global variables
    graph_teen = None
    graph_digit = None
    ties_graph = None

    PYNINI_AVAILABLE = True


def get_ties_graph(deterministic: bool = True):
    """
    Returns two digit transducer, e.g. 
    03 -> o three
    12 -> thirteen
    20 -> twenty
    """
    graph = graph_teen | ties_graph + pynutil.delete("0") | ties_graph + insert_space + graph_digit

    if deterministic:
        graph = graph | pynini.cross("0", "o") + insert_space + graph_digit
    else:
        graph = (
            graph
            | (pynini.cross("0", "oh") | pynini.cross("0", "o") | pynini.cross("0", "zero"))
            + insert_space
            + graph_digit
        )

    return graph.optimize()


def get_hundreds_graph(deterministic: bool = True):
    """
    Returns a four digit transducer which is combination of ties/teen or digits
    (using hundred instead of thousand format), e.g.
    1219 -> twelve nineteen
    3900 -> thirty nine hundred
    """
    graph_ties = get_ties_graph(deterministic)
    graph = (
        graph_ties + insert_space + graph_ties
        | graph_teen + insert_space + pynini.cross("00", "hundred")
        | (graph_teen + insert_space + (ties_graph | pynini.cross("1", "ten")) + pynutil.delete("0s"))
        @ pynini.cdrewrite(pynini.cross("y", "ies") | pynutil.insert("s"), "", "[EOS]", NEMO_SIGMA)
        | pynutil.add_weight(
            graph_digit
            + insert_space
            + pynini.cross("00", "thousand")
            + (pynutil.delete("0") | insert_space + graph_digit),
            weight=-0.001,
        )
        | pynutil.add_weight(
            graph_digit
            + insert_space
            + pynini.cross("000", "thousand")
            + pynini.closure(pynutil.delete(" "), 0, 1)
            + pynini.accep("s"),
            weight=-0.001,
        )
    )
    return graph


def _get_year_graph(deterministic: bool = True):
    """
    Transducer for year, only from 1000 - 2999 e.g.
    1290 -> twelve nineteen
    2000 - 2009 will be verbalized as two thousand.
    """
    graph = get_hundreds_graph(deterministic)
    graph = (
        pynini.union("1", "2")
        + NEMO_DIGIT
        + NEMO_DIGIT
        + NEMO_DIGIT
        + pynini.closure(pynini.cross(" s", "s") | "s", 0, 1)
    ) @ graph
    return graph


class DateFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g. 
        jan. 5, 2012 -> date { month: "january" day: "five" year: "twenty twelve" preserve_order: true }
        jan. 5 -> date { month: "january" day: "five" preserve_order: true }
        5 january 2012 -> date { day: "five" month: "january" year: "twenty twelve" preserve_order: true }
        2012-01-05 -> date { year: "twenty twelve" month: "january" day: "five" }
        2012.01.05 -> date { year: "twenty twelve" month: "january" day: "five" }
        2012/01/05 -> date { year: "twenty twelve" month: "january" day: "five" }
        2012 -> date { year: "twenty twelve" }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        month_graph = pynini.string_file(get_abs_path("data/months/names.tsv")).optimize()
        month_graph |= (TO_LOWER + pynini.closure(NEMO_CHAR)) @ month_graph
        month_abbr_graph = pynini.string_file(get_abs_path("data/months/abbr.tsv")).optimize()
        month_abbr_graph = (
            month_abbr_graph | ((TO_LOWER + pynini.closure(NEMO_CHAR)) @ month_abbr_graph)
        ) + pynini.closure(pynutil.delete("."), 0, 1)
        month_graph |= month_abbr_graph

        # to support all caps names
        names_all_caps = [[x[0].upper()] for x in load_labels(get_abs_path("data/months/names.tsv"))]
        abbr_all_caps = [(x.upper(), y) for x, y in load_labels(get_abs_path("data/months/abbr.tsv"))]
        month_graph |= pynini.string_map(names_all_caps) | (
            pynini.string_map(abbr_all_caps) + pynini.closure(pynutil.delete("."), 0, 1)
        )

        month_numbers_graph = pynini.string_file(get_abs_path("data/months/numbers.tsv")).optimize()
        cardinal_graph = cardinal.graph_hundred_component_at_least_one_none_zero_digit

        year_graph = _get_year_graph(deterministic)

        YEAR_WEIGHT = 0.001
        year_graph_standalone = (
            pynutil.insert("year: \"") + pynutil.add_weight(year_graph, YEAR_WEIGHT) + pynutil.insert("\"")
        )

        month_graph = pynutil.insert("month: \"") + month_graph + pynutil.insert("\"")
        month_numbers_graph = pynutil.insert("month: \"") + month_numbers_graph + pynutil.insert("\"")

        day_graph = (
            pynutil.insert("day: \"")
            + ((pynini.union("1", "2", "3") + NEMO_DIGIT) | NEMO_DIGIT) @ cardinal_graph
            + pynutil.insert("\"")
        )

        two_digit_year = NEMO_DIGIT ** (2) @ (cardinal.single_digits_graph | cardinal_graph)
        two_digit_year = pynutil.insert("year: \"") + two_digit_year + pynutil.insert("\"")

        year_graph = pynutil.insert("year: \"") + year_graph + pynutil.insert("\"")
        graph_year = pynini.closure(pynutil.delete(","), 0, 1) + delete_extra_space + year_graph
        optional_graph_year = pynini.closure(graph_year, 0, 1)
        graph_mdy = month_graph + (
            (delete_extra_space + day_graph) | graph_year | (delete_extra_space + day_graph + graph_year)
        )

        delete_sep = pynutil.delete(pynini.union("-", "/", "."))
        graph_mdy |= (
            month_numbers_graph
            + delete_sep
            + insert_space
            + pynini.closure(pynutil.delete("0"), 0, 1)
            + day_graph
            + delete_sep
            + insert_space
            + (year_graph | two_digit_year)
        )

        graph_dmy = day_graph + delete_extra_space + month_graph + optional_graph_year
        graph_ymd = (
            (year_graph | two_digit_year)
            + delete_sep
            + insert_space
            + month_numbers_graph
            + delete_sep
            + insert_space
            + pynini.closure(pynutil.delete("0"), 0, 1)
            + day_graph
        )

        final_graph = graph_mdy | graph_dmy
        if deterministic:
            final_graph += pynutil.insert(" preserve_order: true")
        else:
            final_graph += pynini.closure(pynutil.insert(" preserve_order: true"), 0, 1)
        final_graph |= graph_ymd | year_graph_standalone

        if not deterministic:
            ymd_to_mdy_graph = None
            mdy_to_dmy_graph = None

            for month in [x[0] for x in load_labels(get_abs_path("data/months/names.tsv"))]:
                for day in [x[0] for x in load_labels(get_abs_path("data/months/days.tsv"))]:
                    ymd_to_mdy_curr = (
                        pynutil.insert("month: \"" + month + "\" day: \"" + day + "\" ")
                        + pynini.accep('year:')
                        + NEMO_SIGMA
                        + pynutil.delete(" month: \"" + month + "\" day: \"" + day + "\"")
                    )

                    # YY-MM-DD -> MM-DD-YY
                    ymd_to_mdy_curr = pynini.compose(final_graph, ymd_to_mdy_curr)
                    ymd_to_mdy_graph = (
                        ymd_to_mdy_curr
                        if ymd_to_mdy_graph is None
                        else pynini.union(ymd_to_mdy_curr, ymd_to_mdy_graph)
                    )

                    mdy_to_dmy_curr = (
                        pynutil.insert("day: \"" + day + "\" month: \"" + month + "\" ")
                        + pynutil.delete("month: \"" + month + "\" day: \"" + day + "\" ")
                        + pynini.accep('year:')
                        + NEMO_SIGMA
                    )

                    # pynini.compose(ymd_to_mdy_curr, mdy_to_dmy_curr) to handle:
                    # YY-MM-DD (input format) -> MM-DD-YY (intermediate ymd_to_mdy_curr representation) -> DD-MM-YY
                    # '2000-01-05' -> 'day: "five" month: "january" year: "two thousand"'
                    # pynini.compose(final_graph, mdy_to_dmy_curr) to handle:
                    # MM-DD-YY (input format) -> DD-MM-YY
                    mdy_to_dmy_curr = pynini.compose(ymd_to_mdy_curr, mdy_to_dmy_curr) | pynini.compose(
                        final_graph, mdy_to_dmy_curr
                    )
                    mdy_to_dmy_graph = (
                        mdy_to_dmy_curr
                        if mdy_to_dmy_graph is None
                        else pynini.union(mdy_to_dmy_curr, mdy_to_dmy_graph)
                    )

            final_graph |= ymd_to_mdy_graph | mdy_to_dmy_graph

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
