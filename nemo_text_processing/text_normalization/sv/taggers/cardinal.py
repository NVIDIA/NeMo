# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.sv.utils import get_abs_path
from pynini.lib import pynutil


zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv")))
digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv")))
teen = pynini.invert(pynini.string_file(get_abs_path("data/numbers/teen.tsv")))
ties = pynini.invert(pynini.string_file(get_abs_path("data/numbers/ties.tsv")))
ett_to_en = pynini.string_map([("ett", "en")])


def make_million(number: str, non_zero_no_one: 'pynini.FstLike', deterministic: bool=True) -> 'pynini.FstLike':
    """
    Helper function for millions/milliards and higher
    Args:
        number: the string of the number
        non_zero_no_one: An fst of digits excluding 0 and 1, to prefix to the number
        deterministic: if True, generate a deterministic fst

    Returns:
        graph: A pynini.FstLike object
    """
    old_orth = number.replace("lj", "lli")
    graph = pynutil.add_weight(pynini.cross("001", number), -0.001)
    if not deterministic:
        graph |= pynutil.add_weight(pynini.cross("001", old_orth), -0.001)
        # 'ett' is usually wrong for these numbers, but it occurs
        for one in ["en", "ett"]:
            graph |= pynutil.add_weight(pynini.cross("001", f"{one} {number}"), -0.001)
            graph |= pynutil.add_weight(pynini.cross("001", f"{one} {old_orth}"), -0.001)
    graph |= non_zero_no_one + pynutil.insert(f" {number}er")
    if not deterministic:
        graph |= non_zero_no_one + pynutil.insert(f" {old_orth}er")
    graph |= pynutil.delete("000")
    graph += insert_space
    return graph


def filter_punctuation(fst: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Helper function for parsing number strings. Converts common cardinal strings (groups of three digits delineated by space)
    and converts to a string of digits:
        "1 000" -> "1000"
    Args:
        fst: Any pynini.FstLike object. Function composes fst onto string parser fst

    Returns:
        fst: A pynini.FstLike object
    """
    exactly_three_digits = NEMO_DIGIT ** 3  # for blocks of three
    up_to_three_digits = pynini.closure(NEMO_DIGIT, 1, 3)  # for start of string

    cardinal_separator = NEMO_SPACE
    cardinal_string = pynini.closure(
        NEMO_DIGIT, 1
    )  # For string w/o punctuation (used for page numbers, thousand series)

    cardinal_string |= (
        up_to_three_digits
        + pynutil.delete(cardinal_separator)
        + pynini.closure(exactly_three_digits + pynutil.delete(cardinal_separator))
        + exactly_three_digits
    )

    return cardinal_string @ fst


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        "1000" ->  cardinal { integer: "tusen" }
        "2 000 000" -> cardinal { integer: "två miljon" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        # Any single digit
        graph_digit = digit
        digits_no_one = (NEMO_DIGIT - "1") @ graph_digit
        both_ones = (pynini.cross("1", "en") | pynini.cross("1", "ett"))
        if deterministic:
            final_digit = digit
        else:
            final_digit = digits_no_one | both_ones

        # Any double digit
        graph_tens = teen
        final_tens = graph_tens
        graph_ties = ties
        if deterministic:
            graph_tens |= graph_ties + (pynutil.delete('0') | graph_digit)
            final_tens = graph_tens
        else:
            graph_ties |= pynini.cross("4", "förtio")
            graph_tens |= pynini.cross("18", "aderton")
            graph_tens |= graph_ties + (pynutil.delete('0') | (graph_digit | pynutil.insert(' ') + graph_digit))
            final_tens |= graph_ties + (pynutil.delete('0') | (final_digit | pynutil.insert(' ') + final_digit))

        hundreds = digits_no_one + pynutil.insert("hundra")
        hundreds |= pynini.cross("1", "hundra")
        if not deterministic:
            hundreds |= pynini.cross("1", "etthundra")
            hundreds |= pynini.cross("1", "ett hundra")
            hundreds |= digit + pynutil.insert(NEMO_SPACE) + pynutil.insert("hundra")

        self.tens = graph_tens.optimize()

        graph_two_digit_non_zero = pynini.union(
            graph_digit, graph_tens, (pynutil.delete("0") + graph_digit)
        )
        if not deterministic:
            graph_two_digit_non_zero |= pynini.union(
                graph_digit, graph_tens, (pynini.cross("0", NEMO_SPACE) + graph_digit)
            )

        self.two_digit_non_zero = graph_two_digit_non_zero.optimize()

        graph_final_two_digit_non_zero = pynini.union(
            final_digit, graph_tens, (pynutil.delete("0") + final_digit)
        )
        if not deterministic:
            graph_final_two_digit_non_zero |= pynini.union(
                final_digit, graph_tens, (pynini.cross("0", NEMO_SPACE) + final_digit)
            )

        self.final_two_digit_non_zero = graph_final_two_digit_non_zero.optimize()

        # Three digit strings
        graph_hundreds = hundreds + pynini.union(
            pynutil.delete("00"), graph_tens, (pynutil.delete("0") + final_digit)
        )
        if not deterministic:
            graph_hundreds |= hundreds + pynini.union(
                pynutil.delete("00"), (graph_tens | pynutil.insert(NEMO_SPACE) + graph_tens), (pynini.cross("0", NEMO_SPACE) + final_digit)
            )

        self.hundreds = graph_hundreds.optimize()

        # For all three digit strings with leading zeroes (graph appends '0's to manage place in string)
        graph_hundreds_component = pynini.union(graph_hundreds, pynutil.delete("0") + graph_tens)

        graph_hundreds_component_at_least_one_non_zero_digit = graph_hundreds_component | (
            pynutil.delete("00") + graph_digit
        )
        graph_hundreds_component_at_least_one_non_zero_digit_no_one = graph_hundreds_component | (
            pynutil.delete("00") + digits_no_one
        )
        self.graph_hundreds_component_at_least_one_non_zero_digit = graph_hundreds_component_at_least_one_non_zero_digit
        self.graph_hundreds_component_at_least_one_non_zero_digit_no_one = graph_hundreds_component_at_least_one_non_zero_digit_no_one.optimize()

        tusen = pynutil.insert("tusen")
        if not deterministic:
            tusen |= pynutil.insert(" tusen")
            tusen |= pynutil.insert("ettusen")
            tusen |= pynutil.insert(" ettusen")
            tusen |= pynutil.insert("ett tusen")
            tusen |= pynutil.insert(" ett tusen")

        graph_thousands_component_at_least_one_non_zero_digit = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_non_zero_digit,
            graph_hundreds_component_at_least_one_non_zero_digit_no_one
            + tusen
            + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
            pynini.cross("001", tusen)
            + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
        )

        graph_thousands_component_at_least_one_non_zero_digit_no_one = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_non_zero_digit_no_one,
            graph_hundreds_component_at_least_one_non_zero_digit_no_one
            + tusen
            + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
            pynini.cross("001", tusen)
            + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
        )

        non_zero_no_one = graph_hundreds_component_at_least_one_non_zero_digit_no_one
        graph_million = make_million("miljon", non_zero_no_one, deterministic)
        graph_milliard = make_million("miljard", non_zero_no_one, deterministic)
        graph_billion = make_million("biljon", non_zero_no_one, deterministic)
        graph_billiard = make_million("biljard", non_zero_no_one, deterministic)
        graph_trillion = make_million("triljon", non_zero_no_one, deterministic)
        graph_trilliard = make_million("triljard", non_zero_no_one, deterministic)

        graph = (
            graph_trilliard
            + graph_trillion
            + graph_billiard
            + graph_billion
            + graph_milliard
            + graph_million
            + (graph_thousands_component_at_least_one_non_zero_digit | pynutil.delete("000000"))
        )

        self.graph = (
            ((NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 0))
            @ pynini.cdrewrite(pynini.closure(pynutil.insert("0")), "[BOS]", "", NEMO_SIGMA)
            @ NEMO_DIGIT ** 24
            @ graph
            @ pynini.cdrewrite(delete_space, "[BOS]", "", NEMO_SIGMA)
            @ pynini.cdrewrite(delete_space, "", "[EOS]", NEMO_SIGMA)
            @ pynini.cdrewrite(
                pynini.cross(pynini.closure(NEMO_WHITE_SPACE, 2), NEMO_SPACE), NEMO_ALPHA, NEMO_ALPHA, NEMO_SIGMA
            )
        )
        self.graph |= zero

        self.graph = filter_punctuation(self.graph).optimize()
        self.graph_en = self.graph @ pynini.cdrewrite(ett_to_en, "", "[EOS]", NEMO_SIGMA)

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
