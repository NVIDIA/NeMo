# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_CHAR, NEMO_SIGMA, NEMO_SPACE, GraphFst
from nemo_text_processing.text_normalization.es.utils import get_abs_path


try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fraction
    "23 4/5" ->
    tokens { fraction { integer: "veintitres" numerator: "quatro" denominator: "quinto" mophosyntactic_features: "ordinal" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, ordinal, deterministic: bool = True): # Change so we're using cardinal from ordinal
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)
        cardinal = ordinal.cardinal
        cardinal_graph = cardinal.graph
        ordinal_graph = ordinal.graph

        ordinal_exceptions = pynini.string_file(get_abs_path("data/fractions/ordinal_exceptions.tsv"))

        # 2-10 are all ordinals
        three_to_ten = pynini.string_map([
			"2",
			"3",
			"4",
			"5",
			"6",
			"7",
			"8",
			"9",
			"10",
		])
        block_three_to_ten = pynutil.delete(three_to_ten) # To block cardinal productions
        if not deterministic: # Multiples of tens are sometimes rendered as ordinals
           three_to_ten |= pynini.string_map([
			   "20",
			   "30",
			   "40",
			   "50",
			   "60",
			   "70",
			   "80",
			   "90",
		   ])      
        graph_three_to_ten = three_to_ten @ ordinal_graph
        graph_three_to_ten @= pynini.cdrewrite(ordinal_exceptions, "", "", NEMO_SIGMA)

        # Higher powers of tens (and multiples) are converted to ordinals.
        higher_powers_of_ten = pynini.string_file(get_abs_path("data/fractions/powers_of_ten.tsv"))
        hundreds = pynini.string_map([
			"100",
			"200",
			"300",
			"400",
			"500",
			"600",
			"700",
			"800",
			"900",
		])
        block_hundreds = pynutil.delete(hundreds) # To block cardinal productions
        graph_hundreds = hundreds @ ordinal_graph

        multiples_of_thousand = ordinal.multiples_of_thousand # So we can have just X milesimo

        graph_higher_powers_of_ten = cardinal.numbers_one_to_one_thousand # Higher powers of ten are ordinals, but their multiples are cardinals
        graph_higher_powers_of_ten += pynini.closure(NEMO_SPACE, 0, 1) + higher_powers_of_ten
        graph_higher_powers_of_ten = cardinal_graph @ graph_higher_powers_of_ten
        graph_higher_powers_of_ten @= pynini.cdrewrite(pynutil.delete("un "), pynini.accep("[BOS]"), pynini.project(higher_powers_of_ten, "output"), NEMO_SIGMA) # we drop 'un' from these ordinals (millionths, not one-millionths

        graph_higher_powers_of_ten = multiples_of_thousand | graph_hundreds | graph_higher_powers_of_ten 
        block_higher_powers_of_ten = pynutil.delete(pynini.project(graph_higher_powers_of_ten, "input")) # For cardinal graph	

        graph_fractions_ordinals = graph_higher_powers_of_ten | graph_three_to_ten
        graph_fractions_ordinals += pynutil.insert("\" morphosyntactic_feautures: \"ordinal\"") # We note the root for processing later

		# Blocking the digits and hundreds from Cardinal graph
        graph_fractions_cardinals = pynini.cdrewrite(block_three_to_ten | block_hundreds, pynini.accep("[BOS]"), pynini.accep("[EOS]"), NEMO_SIGMA)  
        graph_fractions_cardinals @= pynini.cdrewrite(block_higher_powers_of_ten, "", "", NEMO_SIGMA)
        graph_fractions_cardinals @= NEMO_CHAR.plus 
        graph_fractions_cardinals @= pynini.cdrewrite(pynutil.delete("0"), pynini.accep("[BOS]"), pynini.accep("[EOS]"), NEMO_SIGMA) # For some reason empty characters are made '0'
        graph_fractions_cardinals @= cardinal_graph
        graph_fractions_cardinals += pynutil.insert("\" morphosyntactic_feautures: \"cardinal\"") # We note the root for later processing NEMO_CHAR.plus #  blocking these entries to reduce erroneous possibilities in debugging

        graph_denominator = graph_fractions_ordinals | graph_fractions_cardinals

        integer = pynutil.insert("integer_part: \"") + cardinal_graph + pynutil.insert("\"") + pynini.accep(" ")
        numerator = (
            pynutil.insert("numerator: \"") + cardinal_graph + (pynini.cross("/", "\" ") | pynini.cross(" / ", "\" "))
       )

        denominator = pynutil.insert("denominator: \"") + graph_denominator

        self.graph = pynini.closure(integer, 0, 1) + numerator + denominator

        final_graph = self.add_tokens(self.graph)
        if not deterministic: # Also a general form of simply going 'cardinal sobre cardinal'
           self.graph |= cardinal.fst + NEMO_SPACE + cardinal.fst + pynini.cross("/", 'tokens { name: \"sobre\" } }') + cardinal.fst

        self.fst = final_graph.optimize()

