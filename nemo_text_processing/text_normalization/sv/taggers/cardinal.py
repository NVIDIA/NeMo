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

try:
    import pynini
    from pynini.lib import pynutil

    zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv")))
    digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv")))
    teen = pynini.invert(pynini.string_file(get_abs_path("data/numbers/teen.tsv")))
    ties = pynini.invert(pynini.string_file(get_abs_path("data/numbers/ties.tsv")))

    PYNINI_AVAILABLE = True

except (ModuleNotFoundError, ImportError):
    zero = None
    digit = None
    teen = None
    ties = None
    twenties = None
    hundreds = None

    PYNINI_AVAILABLE = False


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
        if deterministic:
            graph_tens |= ties + (pynutil.delete('0') | graph_digit)
            final_tens = graph_tens
        else:
            graph_tens |= ties + (pynutil.delete('0') | (graph_digit | pynutil.insert(' ') + graph_digit))
            final_tens |= ties + (pynutil.delete('0') | (final_digit | pynutil.insert(' ') + final_digit))

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
            pynini.cross("001", "tusen")
            + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
        )

        graph_thousands_component_at_least_one_non_zero_digit_no_one = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_non_zero_digit_no_one,
            graph_hundreds_component_at_least_one_non_zero_digit_no_one
            + tusen
            + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
            pynini.cross("001", "tusen")
            + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
        )

        graph_million = pynutil.add_weight(pynini.cross("001", "miljon"), -0.001)
        if not deterministic:
            graph_million |= pynutil.add_weight(pynini.cross("001", "million"), -0.001)
            graph_million |= pynutil.add_weight(pynini.cross("001", "en miljon"), -0.001)
            graph_million |= pynutil.add_weight(pynini.cross("001", "ett miljon"), -0.001)
            graph_million |= pynutil.add_weight(pynini.cross("001", "en million"), -0.001)
            graph_million |= pynutil.add_weight(pynini.cross("001", "ett million"), -0.001)
        graph_million |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert(" miljoner")
        if not deterministic:
            graph_million |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert(" millioner")
        graph_million |= pynutil.delete("000")
        graph_million += insert_space

        graph_milliard = pynutil.add_weight(pynini.cross("001", "miljard"), -0.001)
        if not deterministic:
            graph_milliard |= pynutil.add_weight(pynini.cross("001", "milliard"), -0.001)
            graph_milliard |= pynutil.add_weight(pynini.cross("001", "en miljard"), -0.001)
            graph_milliard |= pynutil.add_weight(pynini.cross("001", "ett miljard"), -0.001)
            graph_milliard |= pynutil.add_weight(pynini.cross("001", "en milliard"), -0.001)
            graph_milliard |= pynutil.add_weight(pynini.cross("001", "ett milliard"), -0.001)
        graph_milliard |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert(" miljarder")
        if not deterministic:
            graph_milliard |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert(" milliarder")
        graph_milliard |= pynutil.delete("000")
        graph_milliard += insert_space

        graph_billion = pynutil.add_weight(pynini.cross("001", "biljon"), -0.001)
        if not deterministic:
            graph_billion |= pynutil.add_weight(pynini.cross("001", "billion"), -0.001)
            graph_billion |= pynutil.add_weight(pynini.cross("001", "en biljon"), -0.001)
            graph_billion |= pynutil.add_weight(pynini.cross("001", "ett biljon"), -0.001)
            graph_billion |= pynutil.add_weight(pynini.cross("001", "en billion"), -0.001)
            graph_billion |= pynutil.add_weight(pynini.cross("001", "ett billion"), -0.001)
        graph_billion |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert(" biljoner")
        if not deterministic:
            graph_billion |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert(" billioner")
        graph_billion |= pynutil.delete("000")
        graph_billion += insert_space

        graph_billiard = pynutil.add_weight(pynini.cross("001", "biljard"), -0.001)
        if not deterministic:
            graph_billiard |= pynutil.add_weight(pynini.cross("001", "billiard"), -0.001)
            graph_billiard |= pynutil.add_weight(pynini.cross("001", "en biljard"), -0.001)
            graph_billiard |= pynutil.add_weight(pynini.cross("001", "ett biljard"), -0.001)
            graph_billiard |= pynutil.add_weight(pynini.cross("001", "en billiard"), -0.001)
            graph_billiard |= pynutil.add_weight(pynini.cross("001", "ett billiard"), -0.001)
        graph_billiard |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert(" biljarder")
        if not deterministic:
            graph_billiard |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert(" billiarder")
        graph_billiard |= pynutil.delete("000")
        graph_billiard += insert_space

        graph_trillion = pynutil.add_weight(pynini.cross("001", "triljon"), -0.001)
        if not deterministic:
            graph_trillion |= pynutil.add_weight(pynini.cross("001", "trillion"), -0.001)
            graph_trillion |= pynutil.add_weight(pynini.cross("001", "en triljon"), -0.001)
            graph_trillion |= pynutil.add_weight(pynini.cross("001", "ett triljon"), -0.001)
            graph_trillion |= pynutil.add_weight(pynini.cross("001", "en trillion"), -0.001)
            graph_trillion |= pynutil.add_weight(pynini.cross("001", "ett trillion"), -0.001)
        graph_trillion |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert(" triljoner")
        if not deterministic:
            graph_trillion |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert(" trillioner")
        graph_trillion |= pynutil.delete("000")
        graph_trillion += insert_space

        graph_trilliard = pynutil.add_weight(pynini.cross("001", "triljard"), -0.001)
        if not deterministic:
            graph_trilliard |= pynutil.add_weight(pynini.cross("001", "trilliard"), -0.001)
            graph_trilliard |= pynutil.add_weight(pynini.cross("001", "en triljard"), -0.001)
            graph_trilliard |= pynutil.add_weight(pynini.cross("001", "ett triljard"), -0.001)
            graph_trilliard |= pynutil.add_weight(pynini.cross("001", "en trilliard"), -0.001)
            graph_trilliard |= pynutil.add_weight(pynini.cross("001", "ett trilliard"), -0.001)
        graph_trilliard |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert(" triljarder")
        if not deterministic:
            graph_trilliard |= graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert(" trilliarder")
        graph_trilliard |= pynutil.delete("000")
        graph_trilliard += insert_space

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

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
