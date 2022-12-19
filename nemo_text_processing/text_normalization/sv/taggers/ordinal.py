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
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
)
from nemo_text_processing.text_normalization.es.graph_utils import roman_to_int, strip_accent
from nemo_text_processing.text_normalization.sv.utils import get_abs_path
from pynini.lib import pynutil

digit = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/digit.tsv")))
teens = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/teen.tsv")))
ties = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/ties.tsv")))
card_ties = pynini.invert(pynini.string_file(get_abs_path("data/numbers/ties.tsv")))


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal
        	"21:a" -> ordinal { integer: "tjugoförsta" }
    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify")
        cardinal_graph = cardinal.graph

        graph_digit = digit.optimize()
        graph_teens = teens.optimize()
        graph_ties = ties.optimize()
        graph_card_ties = card_ties.optimize()

        graph_tens_component = (
            graph_teens
            | graph_card_ties + graph_digit
            | graph_ties + pynutil.delete('0')
        )

        graph_hundred_component = pynini.union(
            graph_hundreds + pynini.closure(NEMO_SPACE + pynini.union(graph_tens_component, graph_digit), 0, 1),
            graph_tens_component,
            graph_digit,
        )

        digit_or_space = pynini.closure(NEMO_DIGIT | pynini.accep(" "))
        cardinal_format = (NEMO_DIGIT - "0") + pynini.closure(digit_or_space + NEMO_DIGIT, 0, 1)
        a_format = (
            ((pynini.closure(cardinal_format + (NEMO_DIGIT - "1"), 0, 1) + pynini.union("1", "2"))
            | (NEMO_DIGIT - "1") + pynini.union("1", "2")
            | pynini.union("1", "2"))
            + pynutil.delete(pynini.union(":a", ":A"))
        )
        e_format = pynini.closure(
            (NEMO_DIGIT - "1" - "2")
            | (cardinal_format + "1" + NEMO_DIGIT)
            | (cardinal_format + (NEMO_DIGIT - "1") + (NEMO_DIGIT - "1" - "2")),
            1,
        ) + pynutil.delete(pynini.union(":e", ":E"))

        # Need to go up to thousands for fractions
        self.one_to_one_thousand = get_one_to_one_thousand(cardinal_graph)

        thousands = pynini.cross("mil", "milésimo")

        graph_thousands = (
            strip_accent(self.one_to_one_thousand) + NEMO_SPACE + thousands
        )  # Cardinals become prefix for thousands series. Snce accent on the powers of ten we strip accent from leading words
        graph_thousands @= pynini.cdrewrite(delete_space, "", "milésimo", NEMO_SIGMA)  # merge as a prefix
        graph_thousands |= thousands

        self.multiples_of_thousand = (cardinal_graph @ graph_thousands).optimize()

        if (
            not deterministic
        ):  # Formally the words preceding the power of ten should be a prefix, but some maintain word boundaries.
            graph_thousands |= (self.one_to_one_thousand @ graph_hundred_component) + NEMO_SPACE + thousands

        graph_thousands += pynini.closure(NEMO_SPACE + graph_hundred_component, 0, 1)

        ordinal_graph = graph_thousands | graph_hundred_component
        ordinal_graph = cardinal_graph @ ordinal_graph

        if not deterministic:
            # The 10's and 20's series can also be two words
            split_words = pynini.cross("decimo", "décimo ") | pynini.cross("vigesimo", "vigésimo ")
            split_words = pynini.cdrewrite(split_words, "", NEMO_CHAR, NEMO_SIGMA)
            ordinal_graph |= ordinal_graph @ split_words

        # If "octavo" is preceeded by the "o" within string, it needs deletion
        ordinal_graph @= pynini.cdrewrite(pynutil.delete("o"), "", "octavo", NEMO_SIGMA)

        self.graph = ordinal_graph.optimize()

        masc = pynini.accep("gender_masc")
        fem = pynini.accep("gender_fem")
        apocope = pynini.accep("apocope")

        delete_period = pynini.closure(pynutil.delete("."), 0, 1)  # Sometimes the period is omitted f

        accept_masc = delete_period + pynini.cross("º", masc)
        accep_fem = delete_period + pynini.cross("ª", fem)
        accep_apocope = delete_period + pynini.cross("ᵉʳ", apocope)

        # Managing Romanization
        graph_roman = pynutil.insert("integer: \"") + roman_to_int(ordinal_graph) + pynutil.insert("\"")
        if not deterministic:
            # Introduce plural
            plural = pynini.closure(pynutil.insert("/plural"), 0, 1)
            accept_masc += plural
            accep_fem += plural

            # Romanizations have no morphology marker, so in non-deterministic case we provide option for all
            insert_morphology = pynutil.insert(pynini.union(masc, fem)) + plural
            insert_morphology |= pynutil.insert(apocope)
            insert_morphology = (
                pynutil.insert(" morphosyntactic_features: \"") + insert_morphology + pynutil.insert("\"")
            )

            graph_roman += insert_morphology

        else:
            # We insert both genders as default
            graph_roman += pynutil.insert(" morphosyntactic_features: \"gender_masc\"") | pynutil.insert(
                " morphosyntactic_features: \"gender_fem\""
            )

        # Rest of graph
        convert_abbreviation = accept_masc | accep_fem | accep_apocope

        graph = (
            pynutil.insert("integer: \"")
            + ordinal_graph
            + pynutil.insert("\"")
        )
        graph = pynini.union(graph, graph_roman)

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
