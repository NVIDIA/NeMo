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


import pynini
from nemo_text_processing.inverse_text_normalization.es.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, NEMO_SPACE, GraphFst
from pynini.lib import pynutil


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fractions
        e.g. dos quintos -> fraction { numerator: "2" denominator: "5" } 
    This class converts fractions with a denominator up to (and including) 
    "1/999".
    
    Fractions with 4 as their denominator, read as "cuarto(s)", are not
    converted because "room" is also "cuarto", which could cause issues like
        "quiero reservar un cuarto" -> quiero reservar 1/2".
    
    Fractions without a numerator are not converted either to prevent issues
    like:
        "estaba medio dormido" -> "estaba 1/2 dormido"
        
    Args:
        cardinal: CardinalFst
        ordinal: OrdinalFst
    """

    def __init__(self, cardinal: GraphFst, ordinal: GraphFst):
        super().__init__(name="fraction", kind="classify")

        cardinal_graph = cardinal.graph_no_exception
        ordinal_graph = ordinal.graph_masc_num_no_exception

        numbers_read_as_ordinals = pynini.string_file(get_abs_path("data/fractions/numbers_read_as_ordinals.tsv"))
        ordinal_replacements = pynini.string_file(get_abs_path("data/fractions/ordinal_exceptions.tsv"))
        ordinal_replacements = pynini.invert(ordinal_replacements)

        make_singular = pynini.closure(pynutil.delete("s"), 0, 1)

        # process denominators read like ordinals
        #     e.g. "un quinto" -> fraction { numerator: "1" denominator: "5" }

        # exclude cases that are ambiguous or read differently
        ordinal_exceptions = pynini.project(pynini.union("segundo", "tercero", "cuarto"), 'input')

        ordinal_graph |= ordinal_replacements @ ordinal_graph
        ordinal_graph = (pynini.project(ordinal_graph, "input") - ordinal_exceptions.arcsort()) @ ordinal_graph
        ordinal_numbers = (NEMO_SIGMA + make_singular) @ ordinal_graph

        # process other denominators
        #     e.g. "dos dieciochoavo" -> fraction { numerator: "2" denominator: "18" }
        restore_accents = pynini.string_map([('un', 'ún'), ('dos', 'dós'), ('tres', 'trés')])
        restore_accents = NEMO_SIGMA + pynini.closure(pynutil.add_weight(restore_accents, -0.001), 0, 1)
        extend_numbers = NEMO_SIGMA + pynini.closure(pynini.cross('i', "a y "), 0, 1) + NEMO_SIGMA
        delete_endings = NEMO_SIGMA + (pynutil.delete("avo") | pynutil.delete("vo")) + make_singular
        other_denominators = extend_numbers @ delete_endings @ restore_accents @ cardinal_graph

        denominators = ordinal_numbers @ numbers_read_as_ordinals
        denominators |= other_denominators

        # process negative fractions
        #     e.g. "menos dos tercios" -> "fractions { negative: True numerator: "2" denominator: "3" }"
        optional_negative_graph = pynini.closure(pynini.cross("menos", "negative: \"True\"") + NEMO_SPACE, 0, 1)

        # process mixed fractions
        #     e.g. "dos y dos tercios" -> "fractions { integer_part: "2" numerator: "2" denominator: "3" }"
        integer_part_graph = (
            pynutil.insert("integer_part: \"")
            + cardinal_graph
            + pynutil.insert("\"")
            + pynutil.delete(pynini.union(" y", " con"))
        )
        optional_integer_part_graph = pynini.closure(integer_part_graph + NEMO_SPACE, 0, 1)

        numerators_graph = pynutil.insert("numerator: \"") + cardinal_graph + pynutil.insert("\"")

        denominators_graph = pynutil.insert("denominator: \"") + denominators + pynutil.insert("\"")

        proper_fractions = numerators_graph + NEMO_SPACE + denominators_graph
        proper_fractions_with_medio = proper_fractions | (
            pynutil.insert("numerator: \"1\" ") + pynini.cross("medio", "denominator: \"2\"")
        )
        proper_fractions_with_medio = optional_negative_graph + proper_fractions_with_medio

        self.proper_fractions_with_medio = self.add_tokens(proper_fractions_with_medio)

        graph = (
            optional_negative_graph + optional_integer_part_graph + numerators_graph + NEMO_SPACE + denominators_graph
        )

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
