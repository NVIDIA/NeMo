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

from nemo_text_processing.text_normalization.es.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, NEMO_NOT_SPACE, roman_to_int, NEMO_SIGMA, GraphFst, delete_space, NEMO_SPACE, insert_space, NEMO_CHAR
from nemo_text_processing.text_normalization.es.graph_utils import strip_accents

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    PYNINI_AVAILABLE = False


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        	"21.º" -> ordinal { integer: "vigésimo primero" morphosyntactic_features: "gender_masc" }
    This class converts ordinal up to "millesímo" (one thousandth) exclusive.

    Cardinals below ten are not converted (in order to avoid 
    e.g. "1.º hice..." -> "primero hice ...", "2.ª guerra mundial" -> "segunda guerra mundial" 
    and any other odd conversions.)

    This FST also records the ending of the ordinal (called "morphosyntactic_features"):
    either as gender_masc, gender_fem, or apocope. Also introduces plural feature for non-deterministic graphs.

    Args:
        cardinal: CardinalFst
    """

    def __init__(self, cardinal: GraphFst,  deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify")
        self.cardinal = cardinal # Storing since we'll use again for fractions
        cardinal_graph = cardinal.graph 

        graph_digit = pynini.string_file(get_abs_path("data/ordinals/digit.tsv")).invert()
        graph_teens = pynini.string_file(get_abs_path("data/ordinals/teen.tsv")).invert()
        graph_twenties = pynini.string_file(get_abs_path("data/ordinals/twenties.tsv")).invert()
        graph_ties = pynini.string_file(get_abs_path("data/ordinals/ties.tsv")).invert()
        graph_hundreds = pynini.string_file(get_abs_path("data/ordinals/hundreds.tsv")).invert()
        graph_roman = roman_to_int()


        if not deterministic:
			# Some alternative derivations
           graph_teens |= pynini.cross("once", "décimo primero")
           graph_teens |= pynini.cross("doce", "décimo segundo")
           graph_ties |= pynini.cross("sesenta", "setuagésimo")
           graph_digit |= pynini.cross("nueve", "nono")
           graph_digit |= pynini.cross("siete", "sétimo")

        tens = graph_teens | (graph_ties + pynini.closure(pynini.cross(" y ", NEMO_SPACE) + graph_digit, 0, 1)) | graph_twenties

        hundreds = graph_hundreds + pynini.closure(NEMO_SPACE + tens, 0, 1)

        graph_hundred_component = tens | hundreds | graph_digit

        # Need to go up to thousands for fractions.
        numbers_one_to_one_thousand = cardinal.numbers_one_to_one_thousand

        thousands = pynini.cross("mil", "milésimo")
        graph_thousands =  (numbers_one_to_one_thousand @ pynini.cdrewrite(strip_accents, "", "", NEMO_SIGMA)) + NEMO_SPACE + thousands # We accept all cardinals as is. since accent on the powers of ten we drop leading words
        graph_thousands @= pynini.cdrewrite(delete_space, "", "", NEMO_SIGMA) # merge as a prefix
        graph_thousands |= thousands

        self.multiples_of_thousand = (cardinal_graph @ graph_thousands).optimize()

        if not deterministic: # Formally the words preceding the power of ten should be a prefix, but seems to vary   
           graph_thousands |=  (numbers_one_to_one_thousand @ pynini.cdrewrite(strip_accents, "", "", NEMO_SIGMA)) + NEMO_SPACE + thousands

        graph_thousands += (NEMO_SPACE + graph_hundred_component).ques

        ordinal_graph = graph_thousands | graph_hundred_component
        ordinal_graph = cardinal_graph @ ordinal_graph

        if not deterministic:
           # The 10's and 20's series can also be two words
           split_words = pynini.cross("decimo", "décimo ") | pynini.cross("vigesimo", "vigésimo ") 
           split_words = pynini.cdrewrite(split_words, "", "", NEMO_SIGMA)
           ordinal_graph |= (ordinal_graph @ split_words)
		   

        # If "octavo" is preceeded by the "o" within string, it needs deletion
        octavo_rewrite = pynini.cdrewrite(pynutil.delete("o"), "", "octavo", NEMO_SIGMA)

        ordinal_graph @= octavo_rewrite
        self.graph = ordinal_graph.optimize()
        
        # Managing romanization
        delete_period = pynutil.delete(".").ques # Sometimes the period is omitted for abbreviations

        masc = pynini.accep("gender_masc")
        fem = pynini.accep("gender_fem")
        apocope = pynini.accep("apocope")

        accept_masc = delete_period + pynini.cross(pynini.closure(NEMO_CHAR) + "º", masc) # Supposed to be only last letter but conventions vary
        accep_fem = delete_period + pynini.cross(pynini.closure(NEMO_CHAR) + "ª", fem)
        accep_apocope = delete_period + pynini.cross("ᵉʳ", apocope)

		# for romans
        graph_roman = pynutil.insert("integer: \"") + (graph_roman @ ordinal_graph) + pynutil.insert("\"") 
        if not deterministic:  # no morphology marker, so give all
           plural = pynutil.insert("/plural").ques 
           insert_morphology = pynutil.insert(pynini.union(masc, fem)) + plural
           insert_morphology |= pynutil.insert(apocope)
           insert_morphology = pynutil.insert(" morphosyntactic_features: \"") + insert_morphology

           graph_roman += insert_morphology

		# Introduce plural
           accept_masc += plural
           accep_fem += plural

        else:
		# Since ordinals are adjectival and masculine gender is default, we assume apocope
           graph_roman += pynutil.insert(" morphosyntactic_features: \"apocope") 

		# Rest of graph
        convert_abbreviation = accept_masc | accep_fem | accep_apocope

        graph = pynutil.insert("integer: \"") + ordinal_graph + pynutil.insert("\"") + pynutil.insert(" morphosyntactic_features: \"") + convert_abbreviation
        graph = pynini.union(graph, graph_roman) + pynutil.insert("\"")

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()


