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

from nemo_text_processing.inverse_text_normalization.en.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_DIGIT, NEMO_SIGMA, GraphFst, insert_space

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g. 
        one two three one two three five six seven eight -> { number_part: "123-123-5678" }

    This class also support card number and IP format.
        "one two three dot one double three dot o dot four o" -> { number_part: "123.133.0.40"}

        "three two double seven three two one four three two one four three double zero five" ->
            { number_part: 3277 3214 3214 3005}

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst):
        super().__init__(name="telephone", kind="classify")
        # country code, number_part, extension
        digit_to_str = pynini.invert(
            pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
        ).optimize() | pynini.cross("0", pynini.union("o", "oh", "zero"))

        double_digit = pynini.union(
            *[
                pynini.cross(
                    pynini.project(str(i) @ digit_to_str, "output")
                    + pynini.accep(" ")
                    + pynini.project(str(i) @ digit_to_str, "output"),
                    pynutil.insert("double ") + pynini.project(str(i) @ digit_to_str, "output"),
                )
                for i in range(10)
            ]
        )
        double_digit.invert()
        number_part = (
            pynini.closure(digit_to_str + insert_space, 2, 2)
            + digit_to_str
            + pynutil.delete("-")
            + insert_space
            + pynini.closure(digit_to_str + insert_space, 2, 2)
            + digit_to_str
            + pynutil.delete("-")
            + insert_space
            + pynini.closure(digit_to_str + insert_space, 3, 3)
            + digit_to_str
        )
        number_part = (
            pynutil.insert("number_part: \"")
            + pynini.cdrewrite(double_digit, "", "", NEMO_SIGMA) @ pynini.invert(number_part)
            + pynutil.insert("\"")
        )

        str_to_digit = pynini.invert(digit_to_str)
        # to handle cases like "one twenty three"
        two_digit_cardinal = pynini.compose(cardinal.graph_no_exception, NEMO_DIGIT ** 2)
        cardinal_option = (
            (str_to_digit + pynutil.delete(" ") + two_digit_cardinal)
            | two_digit_cardinal
            | (two_digit_cardinal + pynutil.delete(" ") + str_to_digit)
        )

        country_code = (
            pynutil.insert("country_code: \"")
            + pynini.closure(pynini.cross("plus ", "+"), 0, 1)
            + ((pynini.closure(str_to_digit + pynutil.delete(" "), 0, 2) + str_to_digit) | cardinal_option)
            + pynutil.insert("\"")
        )
        optional_country_code = pynini.closure(country_code + pynutil.delete(" ") + insert_space, 0, 1).optimize()
        graph = optional_country_code + number_part

        # card number
        double_digit_to_digit = pynini.compose(double_digit, str_to_digit + pynutil.delete(" ") + str_to_digit)
        card_graph = (double_digit_to_digit | str_to_digit).optimize()
        card_graph = (card_graph + pynini.closure(pynutil.delete(" ") + card_graph)).optimize()
        # reformat card number, group by four
        space_four_digits = insert_space + NEMO_DIGIT ** 4
        card_graph = pynini.compose(card_graph, NEMO_DIGIT ** 4 + space_four_digits ** 3).optimize()
        graph |= pynutil.insert("number_part: \"") + card_graph.optimize() + pynutil.insert("\"")

        # ip
        digit_or_double = pynini.closure(str_to_digit + pynutil.delete(" "), 0, 1) + double_digit_to_digit
        digit_or_double |= double_digit_to_digit + pynini.closure(pynutil.delete(" ") + str_to_digit, 0, 1)
        digit_or_double |= str_to_digit + (pynutil.delete(" ") + str_to_digit) ** (0, 2)
        digit_or_double |= cardinal_option
        digit_or_double = digit_or_double.optimize()

        ip_graph = digit_or_double + (pynini.cross(" dot ", ".") + digit_or_double) ** 3
        graph |= pynutil.insert("number_part: \"") + ip_graph.optimize() + pynutil.insert("\"")

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
