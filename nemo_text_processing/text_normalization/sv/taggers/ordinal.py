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
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
    insert_space
)
from nemo_text_processing.text_normalization.es.graph_utils import roman_to_int, strip_accent
from nemo_text_processing.text_normalization.sv.utils import get_abs_path
from nemo_text_processing.text_normalization.sv.taggers.cardinal import make_million, filter_punctuation
from pynini.lib import pynutil

digit = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/digit.tsv")))
teens = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/teen.tsv")))
ties = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/ties.tsv")))
zero = pynini.invert(pynini.string_file(get_abs_path("data/ordinals/zero.tsv")))
card_ties = pynini.invert(pynini.string_file(get_abs_path("data/numbers/ties.tsv")))
card_digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv")))


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
        graph_card_digit = card_digit.optimize()
        digits_no_one = (NEMO_DIGIT - "1") @ graph_card_digit

        if not deterministic:
            graph_ties |= pynini.cross("4", "förtionde")
            graph_teens |= pynini.cross("18", "adertonde")

        graph_tens_component = (
            graph_teens
            | graph_card_ties + graph_digit
            | graph_ties + pynutil.delete('0')
        )
        self.graph_tens_component = graph_tens_component
        graph_tens = graph_tens_component

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

        suffixed_ordinal = a_format | e_format
        self.suffixed_ordinal = suffixed_ordinal.optimize()

        bare_hundreds = digits_no_one + pynini.cross("00", "hundrade")
        bare_hundreds |= pynini.cross("100", "hundrade")
        if not deterministic:
            bare_hundreds |= pynini.cross("100", "etthundrade")
            bare_hundreds |= pynini.cross("100", "ett hundrade")
            bare_hundreds |= digit + pynutil.insert(NEMO_SPACE) + pynini.cross("00", "hundrade")

        hundreds = digits_no_one + pynutil.insert("hundra")
        hundreds |= pynini.cross("1", "hundra")
        if not deterministic:
            hundreds |= pynini.cross("1", "etthundra")
            hundreds |= pynini.cross("1", "ett hundra")
            hundreds |= digit + pynutil.insert(NEMO_SPACE) + pynutil.insert("hundra")

        graph_hundreds = hundreds + pynini.union(
            graph_tens,
            (pynutil.delete("0") + graph_digit),
        )
        if not deterministic:
            graph_hundreds |= hundreds + pynini.union(
                (graph_teens | pynutil.insert(NEMO_SPACE) + graph_teens), (pynini.cross("0", NEMO_SPACE) + graph_digit)
            )
        graph_hundreds |= bare_hundreds

        graph_hundreds_component = pynini.union(graph_hundreds, pynutil.delete("0") + graph_tens)
        graph_hundreds_component_at_least_one_non_zero_digit = graph_hundreds_component | (
            pynutil.delete("00") + graph_digit
        )
        graph_hundreds_component_at_least_one_non_zero_digit_no_one = graph_hundreds_component | (
            pynutil.delete("00") + digits_no_one
        )

        self.hundreds = graph_hundreds.optimize()

        tusen = pynutil.insert("tusen")
        if not deterministic:
            tusen |= pynutil.insert(" tusen")
            tusen |= pynutil.insert("ettusen")
            tusen |= pynutil.insert(" ettusen")
            tusen |= pynutil.insert("ett tusen")
            tusen |= pynutil.insert(" ett tusen")

        graph_thousands_component_at_least_one_non_zero_digit = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_non_zero_digit,
            cardinal.graph_hundreds_component_at_least_one_non_zero_digit_no_one
            + tusen
            + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
            pynini.cross("001", tusen)
            + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
        )

        graph_thousands_component_at_least_one_non_zero_digit_no_one = pynini.union(
            pynutil.delete("000") + graph_hundreds_component_at_least_one_non_zero_digit_no_one,
            cardinal.graph_hundreds_component_at_least_one_non_zero_digit_no_one
            + tusen
            + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
            pynini.cross("001", tusen)
            + ((insert_space + graph_hundreds_component_at_least_one_non_zero_digit) | pynutil.delete("000")),
        )

        non_zero_no_one = cardinal.graph_hundreds_component_at_least_one_non_zero_digit_no_one
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

        ordinal_endings = pynini.string_map([
            ("ljon", "ljonte"),
            ("ljoner", "ljonte"),
            ("llion", "llionte"),
            ("llioner", "llionte"),
            ("ljard", "ljarte"),
            ("ljarder", "ljarte"),
            ("lliard", "lliarte"),
            ("lliarder", "lliarte"),
            ("tusen", "tusende")
        ])

        self.graph = (
            ((NEMO_DIGIT - "0") + pynini.closure(NEMO_DIGIT, 0))
            @ pynini.cdrewrite(pynini.closure(pynutil.insert("0")), "[BOS]", "", NEMO_SIGMA)
            @ NEMO_DIGIT ** 24
            @ graph
            @ pynini.cdrewrite(delete_space, "[BOS]", "", NEMO_SIGMA)
            @ pynini.cdrewrite(delete_space, "", "[EOS]", NEMO_SIGMA)
            @ pynini.cdrewrite(ordinal_endings, "", "[EOS]", NEMO_SIGMA)
            @ pynini.cdrewrite(
                pynini.cross(pynini.closure(NEMO_WHITE_SPACE, 2), NEMO_SPACE), NEMO_ALPHA, NEMO_ALPHA, NEMO_SIGMA
            )
        )

        cleaned_graph = self.graph
        self.graph |= zero

        self.graph = filter_punctuation(self.graph).optimize()

        self.suffixed_to_words = self.suffixed_ordinal @ self.graph

        tok_graph = (
            pynutil.insert("integer: \"")
            + (cleaned_graph | self.suffixed_to_words)
            + pynutil.insert("\"")
        )

        final_graph = self.add_tokens(tok_graph)
        self.fst = final_graph.optimize()

