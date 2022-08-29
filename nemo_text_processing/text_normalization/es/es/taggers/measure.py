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
    NEMO_NON_BREAKING_SPACE,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    convert_space,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.es.graph_utils import strip_cardinal_apocope
from nemo_text_processing.text_normalization.es.utils import get_abs_path
from pynini.lib import pynutil

unit = pynini.string_file(get_abs_path("data/measures/measurements.tsv"))
unit_complex = pynini.string_file(get_abs_path("data/measures/measurements_complex.tsv"))
unit_plural_fem = pynini.string_file(get_abs_path("data/measures/measurements_plural_fem.tsv"))
unit_plural_masc = pynini.string_file(get_abs_path("data/measures/measurements_plural_masc.tsv"))
math_symbols = pynini.string_file(get_abs_path("data/measures/math_symbols.tsv"))


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure,  e.g.
        "2,4 g" -> measure { cardinal { integer_part: "dos" fractional_part: "cuatro" units: "gramos" preserve_order: true } }
        "1 g" -> measure { cardinal { integer: "un" units: "gramo" preserve_order: true } }
        "1 millón g" -> measure { cardinal { integer: "un quantity: "millón" units: "gramos" preserve_order: true } }
        This class also converts words containing numbers and letters
        e.g. "a-8" —> "a ocho"
        e.g. "1,2-a" —> "uno coma dos a"


    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        fraction: FractionFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(
        self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst, deterministic: bool = True,
    ):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph
        optional_graph_negative = pynini.closure("-", 0, 1)

        unit_singular = unit
        unit_plural = unit_singular @ (unit_plural_fem | unit_plural_masc)

        graph_unit_singular = convert_space(unit_singular)
        graph_unit_plural = convert_space(unit_plural)

        unit_plural = pynutil.insert('units: "') + graph_unit_plural + pynutil.insert('"')

        unit_singular_graph = pynutil.insert('units: "') + graph_unit_singular + pynutil.insert('"')

        unit_complex_plural = unit_complex @ (unit_plural_fem | unit_plural_masc)

        graph_unit_complex = convert_space(unit_complex)
        graph_unit_complex_plural = convert_space(unit_complex_plural)

        graph_unit_denominator = (
            pynini.cross("/", "por") + pynutil.insert(NEMO_NON_BREAKING_SPACE) + graph_unit_singular
        )

        optional_unit_denominator = pynini.closure(
            pynutil.insert(NEMO_NON_BREAKING_SPACE) + graph_unit_denominator, 0, 1,
        )

        complex_unit_singular_graph = (
            pynutil.insert('units: "')
            + ((graph_unit_singular + optional_unit_denominator) | graph_unit_denominator | graph_unit_complex)
            + pynutil.insert('"')
        )

        complex_unit_plural_graph = (
            pynutil.insert('units: "')
            + ((graph_unit_plural + optional_unit_denominator) | graph_unit_denominator | graph_unit_complex_plural)
            + pynutil.insert('"')
        )

        subgraph_decimal = (
            decimal.fst + insert_space + pynini.closure(NEMO_SPACE, 0, 1) + (unit_plural | complex_unit_plural_graph)
        )

        subgraph_cardinal = (
            (optional_graph_negative + (NEMO_SIGMA - "1")) @ cardinal.fst
            + insert_space
            + pynini.closure(delete_space, 0, 1)
            + (unit_plural | complex_unit_plural_graph)
        )

        subgraph_cardinal |= (
            (optional_graph_negative + pynini.accep("1")) @ cardinal.fst
            + insert_space
            + pynini.closure(delete_space, 0, 1)
            + (unit_plural | complex_unit_singular_graph)
        )

        """transform "1 1/2 km" ->  measure { cardinal { integer: "uno" } 
                                    fraction { numerator: "un" denominator: "medio" morphosyntactic_features: "ordinal" } 
                                    units: "kilómetro" } }
        """
        int_before_fraction = pynutil.add_weight(
            (optional_graph_negative + pynini.accep("1")) @ cardinal.fst
            + insert_space
            + pynini.closure(delete_space, 0, 1),
            -0.002,
        )

        subgraph_fraction = pynutil.add_weight(
            pynini.closure(int_before_fraction, 0, 1)
            + fraction.fst
            + insert_space
            + pynini.closure(delete_space, 0, 1)
            + unit_singular_graph,
            -0.002,
        )

        """transform "2 1/2 km" ->  measure { cardinal { integer: "dos" } 
                                    fraction { numerator: "un" denominator: "medio" morphosyntactic_features: "ordinal" } 
                                    units: "kilómetros" } }
        """
        ints_before_fraction = (
            (optional_graph_negative + (NEMO_SIGMA - "1")) @ cardinal.fst
            + insert_space
            + pynini.closure(delete_space, 0, 1)
        )

        subgraph_fraction |= pynutil.add_weight(
            ints_before_fraction + fraction.fst + insert_space + pynini.closure(delete_space, 0, 1) + unit_plural,
            -0.002,
        )

        """transform "2 1/2 k/h" -> measure { fraction { integer_part: "dos" 
                                                         numerator: "un" 
                                                         denominator: "medio" 
                                                         morphosyntactic_features: "ordinal" } 
                                              units: "kilómetros" } }
        """
        proper_fraction = pynini.closure(NEMO_DIGIT) + pynini.accep("/") + pynini.closure(NEMO_DIGIT)
        subgraph_complex_units = pynutil.add_weight(
            (proper_fraction @ fraction.fst)
            + insert_space
            + pynini.closure(delete_space, 0, 1)
            + complex_unit_singular_graph
            + pynutil.insert(' preserve_order: true'),
            -0.001,
        )

        subgraph_complex_units |= (
            ((NEMO_SIGMA - proper_fraction) @ fraction.fst)
            + insert_space
            + pynini.closure(delete_space, 0, 1)
            + complex_unit_plural_graph
            + pynutil.insert(' preserve_order: true')
        )

        decimal_times = (
            pynutil.insert("decimal { ")
            + decimal.final_graph_wo_negative
            + pynutil.insert(' } units: "')
            + pynini.union("x", "X")
            + pynutil.insert('"')
        )

        cardinal_times = (
            pynutil.insert('cardinal { integer: "')
            + strip_cardinal_apocope(cardinal_graph)
            + pynutil.insert('" } units: "')
            + pynini.union("x", "X")
            + pynutil.insert('"')
        )

        cardinal_dash_alpha = (
            pynutil.insert('cardinal { integer: "')
            + strip_cardinal_apocope(cardinal_graph)
            + pynutil.delete("-")
            + pynutil.insert('" } units: "')
            + pynini.closure(NEMO_ALPHA, 1)
            + pynutil.insert('"')
        )

        decimal_dash_alpha = (
            pynutil.insert("decimal { ")
            + decimal.final_graph_wo_negative
            + pynutil.delete("-")
            + pynutil.insert(' } units: "')
            + pynini.closure(NEMO_ALPHA, 1)
            + pynutil.insert('"')
        )

        alpha_dash_cardinal = (
            pynutil.insert('units: "')
            + pynini.closure(NEMO_ALPHA, 1)
            + pynutil.delete("-")
            + pynutil.insert('"')
            + pynutil.insert(' cardinal { integer: "')
            + cardinal_graph
            + pynutil.insert('" } preserve_order: true')
        )

        alpha_dash_decimal = (
            pynutil.insert('units: "')
            + pynini.closure(NEMO_ALPHA, 1)
            + pynutil.delete("-")
            + pynutil.insert('"')
            + pynutil.insert(" decimal { ")
            + decimal.final_graph_wo_negative
            + pynutil.insert(" } preserve_order: true")
        )

        delimiter = pynini.accep(" ") | pynutil.insert(" ")
        math_numbers = cardinal_graph | pynutil.add_weight(cardinal_graph @ pynini.cross("un", "uno"), -1) | NEMO_ALPHA
        math = (
            math_numbers
            + pynini.closure(delimiter + math_symbols + delimiter + math_numbers, 1)
            + delimiter
            + pynini.cross("=", "es igual a")
            + delimiter
            + math_numbers
        )

        math |= (
            math_numbers
            + delimiter
            + pynini.cross("=", "es igual a")
            + math_numbers
            + pynini.closure(delimiter + math_symbols + delimiter + math_numbers, 1)
        )

        math = (
            pynutil.insert("units: \"math\" cardinal { integer: \"")
            + math
            + pynutil.insert("\" } preserve_order: true")
        )

        final_graph = (
            subgraph_decimal
            | subgraph_cardinal
            | cardinal_dash_alpha
            | subgraph_complex_units
            | alpha_dash_cardinal
            | decimal_dash_alpha
            | subgraph_fraction
            | decimal_times
            | cardinal_times
            | alpha_dash_decimal
            | math
        )
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
