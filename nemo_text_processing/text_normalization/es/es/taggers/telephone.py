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
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.es.graph_utils import ones
from nemo_text_processing.text_normalization.es.utils import get_abs_path
from pynini.lib import pynutil

graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv"))
graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv"))
graph_twenties = pynini.string_file(get_abs_path("data/numbers/twenties.tsv"))


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone numbers, e.g.
        123-123-5678 -> { number_part: "uno dos tres uno dos tres cinco seis siete ocho" }.
        In Spanish, digits are generally read individually, or as 2-digit numbers,
        eg. "123" = "uno dos tres",
            "1234" = "doce treinta y cuatro".
        This will verbalize sequences of 10 (3+3+4 e.g. 123-456-7890).
        9 (3+3+3 e.g. 123-456-789) and 8 (4+4 e.g. 1234-5678) digits.

        (we ignore more complicated cases such as "doscientos y dos" or "tres nueves").

    Args:
		deterministic: if True will provide a single transduction option,
			for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify")

        # create `single_digits` and `double_digits` graphs as these will be
        # the building blocks of possible telephone numbers
        single_digits = pynini.invert(graph_digit).optimize() | pynini.cross("0", "cero")
        single_digits @= pynini.cdrewrite(pynini.cross(ones, "uno"), "", "", NEMO_SIGMA)

        # Any double digit
        teen = pynini.invert(graph_teen)
        ties = pynini.invert(graph_ties)
        twenties = pynini.invert(graph_twenties)

        double_digits = teen
        double_digits |= ties + (pynutil.delete('0') | (pynutil.insert(" y ") + single_digits))
        double_digits |= twenties

        # define separators
        separators = pynini.union("-", " ", ".")
        delete_separator = pynini.closure(pynutil.delete(separators), 0, 1)

        # process country codes as '+1' -> country_code: "one"
        triple_number = pynini.closure(single_digits + insert_space, 2, 2) + single_digits
        country_code = pynini.closure(pynini.cross("+", "más "), 0, 1) + (
            single_digits | double_digits | triple_number
        )

        # add ip and telephone prompts to this tag (as is in EN)
        ip_prompts = pynini.string_file(get_abs_path("data/telephone/ip_prompt.tsv"))
        telephone_prompts = pynini.string_file(get_abs_path("data/telephone/telephone_prompt.tsv"))
        tel_prompt_sequence = telephone_prompts + NEMO_SPACE + pynini.closure(country_code, 0, 1)

        country_code_graph = (
            pynutil.insert("country_code: \"")
            + (country_code | ip_prompts | tel_prompt_sequence)
            + delete_separator
            + pynutil.insert("\"")
        )

        # process IP addresses
        digit_to_str_graph = single_digits + pynini.closure(pynutil.insert(" ") + single_digits, 0, 2)
        ip_graph = digit_to_str_graph + (pynini.cross(".", " punto ") + digit_to_str_graph) ** 3

        # process area codes with or without parentheses i.e. "212" or (212)
        area_code = (
            pynini.closure(pynutil.delete("("), 0, 1)
            + pynini.closure(single_digits + insert_space, 3, 3)
            + pynini.closure(pynutil.delete(")"), 0, 1)
        )

        # process extensions
        delete_ext = pynini.closure(pynutil.delete("ext."), 0, 1)
        ext_graph = (
            pynutil.insert("extension: \"")
            + delete_separator
            + delete_ext
            + delete_separator
            + pynutil.insert("extensión ")
            + pynini.closure(single_digits + insert_space, 1, 3)
            + single_digits
            + pynutil.insert("\"")
        )

        # define `ten_digit_graph`, `nine_digit_graph`, `eight_digit_graph`
        # which produces telephone numbers spoken (1) only with single digits,
        # or (2) spoken with double digits (and sometimes single digits)

        # 10-digit option (1): all single digits
        ten_digit_graph = (
            area_code
            + delete_separator
            + pynini.closure(single_digits + insert_space, 3, 3)
            + delete_separator
            + pynini.closure(single_digits + insert_space, 3, 3)
            + single_digits
        )

        # 9-digit option (1): all single digits
        nine_digit_graph = (
            area_code
            + delete_separator
            + pynini.closure(single_digits + insert_space, 3, 3)
            + delete_separator
            + pynini.closure(single_digits + insert_space, 2, 2)
            + single_digits
        )

        # 8-digit option (1): all single digits
        eight_digit_graph = (
            pynini.closure(area_code, 0, 1)
            + delete_separator
            + pynini.closure(single_digits + insert_space, 4, 4)
            + delete_separator
            + pynini.closure(single_digits + insert_space, 3, 3)
            + single_digits
        )

        if not deterministic:
            # 10-digit option (2): (1+2) + (1+2) + (2+2) digits
            ten_digit_graph |= (
                pynini.closure(single_digits + insert_space + double_digits + insert_space + delete_separator, 2, 2)
                + double_digits
                + insert_space
                + double_digits
            )

            # 9-digit option (2): (1+2) + (1+2) + (1+2) digits
            nine_digit_graph |= (
                pynini.closure(single_digits + insert_space + double_digits + insert_space + delete_separator, 2, 2)
                + single_digits
                + insert_space
                + double_digits
            )

            # 8-digit option (2): (2+2) + (2+2) digits
            eight_digit_graph |= (
                double_digits
                + insert_space
                + double_digits
                + insert_space
                + delete_separator
                + double_digits
                + insert_space
                + double_digits
            )

        # handle numbers with letters like "1-800-go-u-haul"
        num_letter_area_code = area_code @ pynini.cross("ocho cero cero ", "ochocientos ")
        number_word = pynini.closure(single_digits | NEMO_ALPHA, 1, 8)
        number_words = pynini.closure(number_word + pynini.cross(separators, " "), 0, 2) + number_word

        nums_w_letters_graph = pynutil.add_weight(num_letter_area_code + delete_separator + number_words, 0.01)

        number_part = pynini.union(
            ten_digit_graph, nine_digit_graph, eight_digit_graph, nums_w_letters_graph, ip_graph
        )
        number_part = pynutil.insert("number_part: \"") + number_part + pynutil.insert("\"")

        graph = (
            pynini.closure(country_code_graph + delete_separator + insert_space, 0, 1)
            + number_part
            + pynini.closure(delete_separator + insert_space + ext_graph, 0, 1)
        )
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
