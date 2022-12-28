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
from nemo_text_processing.text_normalization.hu.utils import get_abs_path
from pynini.lib import pynutil

zero = pynini.invert(pynini.string_file(get_abs_path("data/number/zero.tsv")))
digit = pynini.invert(pynini.string_file(get_abs_path("data/number/digit.tsv")))
digit_inline = pynini.invert(pynini.string_file(get_abs_path("data/number/digit_inline.tsv")))
tens = pynini.invert(pynini.string_file(get_abs_path("data/number/tens.tsv")))
tens_inline = pynini.invert(pynini.string_file(get_abs_path("data/number/tens_inline.tsv")))
delete_hyphen = pynutil.delete(pynini.closure("-"))


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
        "1000" ->  cardinal { integer: "ezer" }
        "9999" -> cardinal { integer: "kilencezer-kilencszázkilencvenkilenc" }
        "2000000" -> cardinal { integer: "kétmillió" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        # Any single digit
        graph_digit = digit
        graph_zero = zero
        digits_inline_no_one = (NEMO_DIGIT - "1") @ digit_inline
        digits_no_one = (NEMO_DIGIT - "1") @ digit
        if not deterministic:
            graph_digit |= pynini.cross("2", "két")
            digits_inline_no_one |= pynini.cross("2", "kettő")

        insert_hyphen = pynutil.insert("-")
        # in the non-deterministic case, add an optional space
        if not deterministic:
            insert_hyphen |= pynini.closure(pynutil.insert(" "), 0, 1)

        # Any double digit
        graph_tens = (tens_inline + digit) | tens

        self.tens = graph_tens.optimize()

        self.two_digit_non_zero = pynini.union(
            graph_digit, graph_tens, (pynutil.delete("0") + digit)
        ).optimize()

        base_hundreds = pynini.union(
            pynini.cross("1", "száz"),
            digits_inline_no_one + pynutil.insert("száz")
        )
        if not deterministic:
            base_hundreds |= pynini.cross("1", "egyszáz")

        hundreds = pynini.union(
            pynini.cross("100", "száz"),
            pynini.cross("1", "száz") + graph_tens,
            digits_inline_no_one + pynini.cross("00", "száz"),
            digits_inline_no_one + pynutil.insert("száz") + graph_tens
        )
        if not deterministic:
            hundreds |= pynini.union(
                pynini.cross("100", "egyszáz"),
                pynini.cross("1", "egyszáz") + graph_tens
            )

        # Three digit strings
        graph_hundreds = base_hundreds + pynini.union(
            pynutil.delete("00"), graph_tens, (pynutil.delete("0") + graph_digit)
        )

        self.hundreds = graph_hundreds.optimize()

        # For all three digit strings with leading zeroes (graph appends '0's to manage place in string)
        graph_hundreds_component = pynini.union(graph_hundreds, pynutil.delete("0") + graph_tens)

        graph_hundreds_component_at_least_one_non_zero_digit = graph_hundreds_component | (
            pynutil.delete("00") + graph_digit
        )
        # Needed?
        graph_hundreds_component_at_least_one_non_zero_digit_no_one = graph_hundreds_component | (
            pynutil.delete("00") + digits_no_one
        )

        one_thousand = pynini.union(
            pynini.cross("1000", "ezer"),
            pynini.cross("10", "ezer") + graph_tens,
            pynini.cross("1", "ezer") + base_hundreds
        )
        if not deterministic:
            one_thousand |= pynini.union(
                pynini.cross("1000", "egyezer"),
                pynini.cross("10", "egyezer") + graph_tens,
                pynini.cross("1", "egyezer") + base_hundreds
            )

        # FIXME: 1-9000 only
        # FIXME: missing hundreds + digits: Error: No valid output with given input: '1111'
        # FIXME: missing digits: Error: No valid output with given input: '1001'
        other_thousands = pynini.union(
            digits_inline_no_one + pynini.cross("000", "ezer"),
            digits_inline_no_one + pynini.cross("0", "ezer") + insert_hyphen + graph_tens,
            digits_inline_no_one + pynutil.insert("ezer") + insert_hyphen + base_hundreds
        )

        self.thousands = (one_thousand | other_thousands).optimize()

        # graph_thousands_component_at_least_one_non_zero_digit = pynini.union(
        #     pynutil.delete("000") + graph_hundreds_component_at_least_one_non_zero_digit,
            
        #     + pynutil.insert(" mil")
        #     + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
        #     pynini.cross("001", "mil")
        #     + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
        # )

        # graph_thousands_component_at_least_one_none_zero_digit_no_one = pynini.union(
        #     pynutil.delete("000"),
            
        #     + pynutil.insert(" mil")
        #     + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
        #     pynini.cross("001", "mil")
        #     + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
        # )
        graph_thousands_component_at_least_one_non_zero_digit = one_thousand | other_thousands
        graph_thousands_component_at_least_one_none_zero_digit_no_one = other_thousands
        self.graph_thousands_component_at_least_one_non_zero_digit = graph_thousands_component_at_least_one_non_zero_digit
        self.graph_thousands_component_at_least_one_none_zero_digit_no_one = graph_thousands_component_at_least_one_none_zero_digit_no_one

        graph_million = pynutil.add_weight(pynini.cross("001", "millió"), -0.001)
        graph_million |= (graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert("millió"))
        if not deterministic:
            graph_million |= pynutil.add_weight(pynini.cross("001", "egymillió"), -0.001)
        graph_million |= pynutil.delete("000")
        graph_million += insert_hyphen

        graph_milliard = pynutil.add_weight(pynini.cross("001", "milliárd"), -0.001)
        graph_milliard |= (graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert("milliárd"))
        if not deterministic:
            graph_million |= pynutil.add_weight(pynini.cross("001", "egymilliárd"), -0.001)
        graph_milliard |= pynutil.delete("000")
        graph_milliard += insert_hyphen

        graph_billion = pynutil.add_weight(pynini.cross("001", "billió"), -0.001)
        graph_billion |= (graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert("billió"))
        if not deterministic:
            graph_million |= pynutil.add_weight(pynini.cross("001", "egybillió"), -0.001)
        graph_billion |= pynutil.delete("000")
        graph_billion += insert_hyphen

        graph_billiard = pynutil.add_weight(pynini.cross("001", "billiárd"), -0.001)
        graph_billiard |= (graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert("billiárd"))
        if not deterministic:
            graph_million |= pynutil.add_weight(pynini.cross("001", "egybilliárd"), -0.001)
        graph_billiard |= pynutil.delete("000")
        graph_billiard += insert_hyphen

        graph_trillion = pynutil.add_weight(pynini.cross("001", "trillió"), -0.001)
        graph_trillion |= (graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert("trillió"))
        if not deterministic:
            graph_million |= pynutil.add_weight(pynini.cross("001", "egytrillió"), -0.001)
        graph_trillion |= pynutil.delete("000")
        graph_trillion += insert_hyphen

        graph_trilliard = pynutil.add_weight(pynini.cross("001", "trilliárd"), -0.001)
        graph_trilliard |= (graph_hundreds_component_at_least_one_non_zero_digit_no_one + pynutil.insert("trilliárd"))
        if not deterministic:
            graph_million |= pynutil.add_weight(pynini.cross("001", "egytrilliárd"), -0.001)
        graph_trilliard |= pynutil.delete("000")
        graph_trilliard += insert_hyphen

        graph = (
            graph_trilliard
            + graph_trillion
            + graph_billiard
            + graph_billion
            + graph_milliard
            + graph_million
            + (graph_thousands_component_at_least_one_non_zero_digit | pynutil.delete("000"))
        )

        self.graph = (
            ((NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 0))
            @ pynini.cdrewrite(pynini.closure(pynutil.insert("0")), "[BOS]", "", NEMO_SIGMA)
            @ NEMO_DIGIT ** 24
            @ graph
            @ pynini.cdrewrite(delete_space, "[BOS]", "", NEMO_SIGMA)
            @ pynini.cdrewrite(delete_space, "", "[EOS]", NEMO_SIGMA)
            @ pynini.cdrewrite(delete_hyphen, "[BOS]", "", NEMO_SIGMA)
            @ pynini.cdrewrite(delete_hyphen, "", "[EOS]", NEMO_SIGMA)
            @ pynini.cdrewrite(
                pynini.cross(pynini.closure(NEMO_WHITE_SPACE, 2), NEMO_SPACE), NEMO_ALPHA, NEMO_ALPHA, NEMO_SIGMA
            )
        )
        self.graph |= graph_zero

        self.graph = filter_punctuation(self.graph).optimize()

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
