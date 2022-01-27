# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
	NEMO_SPACE,
    GraphFst,
    NEMO_WHITE_SPACE,
    delete_space,
    insert_space,
	NEMO_ALPHA
)
from nemo_text_processing.text_normalization.es.graph_utils import cardinal_seperator
from nemo_text_processing.text_normalization.es.utils import get_abs_path

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False

def get_one_to_one_thousand(cardinal):
	numbers = [str(_) for _ in range(1,1000)]
	graph = pynini.string_map(numbers) @ cardinal
	graph = pynini.project(graph, "output")
	return graph.optimize()

def filter_punctuation(fst):
	exactly_three_digits = NEMO_DIGIT ** 3  # for blocks of three
	up_to_three_digits = pynini.closure(NEMO_DIGIT, 1, 3)  # for start of string

	cardinal_string = pynini.closure(NEMO_DIGIT, 1) # For string w/o punctuation (used for page numbers, multiples of thousand)

	cardinal_string |= (
		up_to_three_digits
		+ pynutil.delete(cardinal_seperator)
		+ pynini.closure(exactly_three_digits + pynutil.delete(cardinal_seperator))
		+ exactly_three_digits
	)

	return cardinal_string @ fst


class GenderAlignment:
    def __init__(self):
        fem_hundreds = pynini.cross("ientos", "ientas")

        fem_ones = pynini.cross("un", "una") | pynini.cross("ún", "una")
        masc_ones = pynini.string_map([("un", "uno"), ("ún", "uno"), "un", "ún",])

        fem_align = pynini.cdrewrite(fem_hundreds, "", "", NEMO_SIGMA)
        fem_align @= pynini.cdrewrite(fem_ones, "", "[EOS]", NEMO_SIGMA)

        masc_split_ones = pynini.cdrewrite(masc_ones, "", "[EOS]", NEMO_SIGMA)

        self.fst = (fem_align | masc_split_ones).optimize()


class CardinalFst(GraphFst):
    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        graph_teen = pynini.string_file(get_abs_path("data/numbers/teen.tsv")).invert()
        graph_ties = pynini.string_file(get_abs_path("data/numbers/ties.tsv")).invert()
        graph_twenties = pynini.string_file(get_abs_path("data/numbers/twenties.tsv")).invert()
        graph_hundreds = pynini.string_file(get_abs_path("data/numbers/hundreds.tsv")).invert()

        # Any single digit
        digits =  graph_digit
        digits_no_one = (NEMO_DIGIT - "1") @ digits

        # Any double digit
        tens = graph_teen
        tens |= graph_ties + (pynutil.delete('0') | (pynutil.insert(" y ") + graph_digit))
        tens |= graph_twenties

        self.tens = tens.optimize()

        # Three digit strings
        hundreds = graph_hundreds
        hundreds += pynini.union(pynutil.delete("00"), (insert_space + tens), (pynini.cross("0", NEMO_SPACE) + digits))
        hundreds |= pynini.cross("100", "cien")
        hundreds |= pynini.cross("1", "ciento") + insert_space + pynini.union(tens, pynutil.delete("0") + digits)

        self.hundreds = hundreds.optimize()

        # For all three digit strings with leading zeroes (we insert them in our graph prior)
        graph_hundreds_component = pynini.union(hundreds, pynutil.delete("0") + tens)

        graph_hundreds_component_at_least_one_none_zero_digit = graph_hundreds_component | (pynutil.delete("00") + digits)
        graph_hundreds_component_at_least_one_none_zero_digit_no_one = graph_hundreds_component | (pynutil.delete("00") + digits_no_one)        

        # Larger numbers (manage spaces here)
        graph_thousands = pynini.cross("001", "mil")
        graph_thousands |= graph_hundreds_component_at_least_one_none_zero_digit_no_one + pynutil.insert(" mil")
        graph_thousands |= pynutil.delete("000")
        graph_thousands += insert_space

        graph_million = pynutil.add_weight(pynini.cross("001", "un millón"), -0.001)
        graph_million |= graph_hundreds_component_at_least_one_none_zero_digit_no_one + pynutil.insert(" millones")
        graph_million |= pynutil.delete("000")
        graph_million += insert_space

        graph_mil_million = pynutil.add_weight(pynini.cross("001", "mil millones"), -0.001)
        graph_mil_million |= graph_hundreds_component_at_least_one_none_zero_digit_no_one + pynutil.insert(
            " mil millones"
        )
        graph_mil_million |= pynutil.delete("000")
        graph_mil_million += insert_space

        graph_billion = pynutil.add_weight(pynini.cross("001", "un billón"), -0.001)
        graph_billion |= graph_hundreds_component_at_least_one_none_zero_digit_no_one + pynutil.insert(" billones")
        graph_billion |= pynutil.delete("000")
        graph_billion += insert_space

        graph_mil_billion = pynutil.add_weight(pynini.cross("001", "mil billones"), -0.001)
        graph_mil_billion |= graph_hundreds_component_at_least_one_none_zero_digit_no_one + pynutil.insert(
            " mil billones"
        )
        graph_mil_billion |= pynutil.delete("000")
        graph_mil_billion += insert_space

        graph_trillion = pynutil.add_weight(pynini.cross("001", "un trillón"), -0.001)
        graph_trillion |= graph_hundreds_component_at_least_one_none_zero_digit_no_one + pynutil.insert(" trillones")
        graph_trillion |= pynutil.delete("000")
        graph_trillion += insert_space

        graph_mil_trillion = pynutil.add_weight(pynini.cross("001", "mil trillones"), -0.001)
        graph_mil_trillion |= graph_hundreds_component_at_least_one_none_zero_digit_no_one + pynutil.insert(
            " mil trillones"
        )
        graph_mil_trillion |= pynutil.delete("000")
        graph_mil_trillion += insert_space

        graph = (
            graph_mil_trillion
            + graph_trillion
            + graph_mil_billion
            + graph_billion
            + graph_mil_million
            + graph_million
            + graph_thousands
            + (graph_hundreds_component_at_least_one_none_zero_digit | pynutil.delete("000"))
        )

        self.graph = (
            (NEMO_DIGIT - "0" + pynini.closure(NEMO_DIGIT, 0))
            @ pynini.cdrewrite(pynini.closure(pynutil.insert("0")), "[BOS]", "", NEMO_SIGMA)
            @ NEMO_DIGIT ** 24
            @ graph 
            @ pynini.cdrewrite(delete_space, "[BOS]", "", NEMO_SIGMA)
            @ pynini.cdrewrite(delete_space, "", "[EOS]", NEMO_SIGMA)
            @ pynini.cdrewrite(pynini.cross(pynini.closure(NEMO_WHITE_SPACE, 2), NEMO_SPACE), NEMO_ALPHA, NEMO_ALPHA, NEMO_SIGMA)
        )
        self.graph |= graph_zero

        self.graph = filter_punctuation(self.graph).optimize()

        self.numbers_one_to_one_thousand = get_one_to_one_thousand(self.graph)

        optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        if not deterministic:
           derivations = pynutil.insert(" morphosyntactic_features: ")  # Allows verbalizer to apply gender allignment
           derivations += pynutil.insert("\"gender_fem\"") | pynutil.insert("\"no_apocope\"")
           final_graph += derivations.ques
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()