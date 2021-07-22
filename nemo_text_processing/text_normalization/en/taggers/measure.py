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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NON_BREAKING_SPACE,
    NEMO_SIGMA,
    NEMO_SPACE,
    SINGULAR_TO_PLURAL,
    GraphFst,
    convert_space,
    delete_space,
)
from nemo_text_processing.text_normalization.en.taggers.ordinal import OrdinalFst as OrdinalTagger
from nemo_text_processing.text_normalization.en.utils import get_abs_path
from nemo_text_processing.text_normalization.en.verbalizers.ordinal import OrdinalFst as OrdinalVerbalizer

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure, suppletive aware, e.g. 
        -12kg -> measure { negative: "true" cardinal { integer: "twelve" } units: "kilograms" }
        1kg -> measure { cardinal { integer: "one" } units: "kilogram" }
        .5kg -> measure { decimal { fractional_part: "five" } units: "kilograms" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        fraction: FractionFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, fraction: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph

        if not deterministic:
            cardinal_graph |= cardinal.range_graph

        graph_unit = pynini.string_file(get_abs_path("data/measurements.tsv"))
        graph_unit_plural = convert_space(graph_unit @ SINGULAR_TO_PLURAL)
        graph_unit = convert_space(graph_unit)
        optional_graph_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        graph_unit2 = pynini.cross("/", "per") + delete_space + pynutil.insert(NEMO_NON_BREAKING_SPACE) + graph_unit

        optional_graph_unit2 = pynini.closure(
            delete_space + pynutil.insert(NEMO_NON_BREAKING_SPACE) + graph_unit2, 0, 1,
        )

        unit_plural = (
            pynutil.insert("units: \"")
            + (graph_unit_plural + optional_graph_unit2 | graph_unit2)
            + pynutil.insert("\"")
        )

        unit_singular = (
            pynutil.insert("units: \"") + (graph_unit + optional_graph_unit2 | graph_unit2) + pynutil.insert("\"")
        )

        subgraph_decimal = (
            pynutil.insert("decimal { ")
            + optional_graph_negative
            + decimal.final_graph_wo_negative
            + delete_space
            + pynutil.insert(" } ")
            + unit_plural
        )

        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + ((NEMO_SIGMA - "1") @ cardinal_graph)
            + delete_space
            + pynutil.insert("\"")
            + pynutil.insert(" } ")
            + unit_plural
        )

        subgraph_cardinal |= (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + pynini.cross("1", "one")
            + delete_space
            + pynutil.insert("\"")
            + pynutil.insert(" } ")
            + unit_singular
        )

        cardinal_dash_alpha = (
            pynutil.insert("cardinal { integer: \"")
            + cardinal_graph
            + pynini.cross('-', '')
            + pynutil.insert("\" } units: \"")
            + pynini.closure(NEMO_ALPHA, 1)
            + pynutil.insert("\"")
        )

        alpha_dash_cardinal = (
            pynutil.insert("units: \"")
            + pynini.closure(NEMO_ALPHA, 1)
            + pynini.cross('-', '')
            + pynutil.insert("\"")
            + pynutil.insert(" cardinal { integer: \"")
            + cardinal_graph
            + pynutil.insert("\" } preserve_order: true")
        )

        decimal_dash_alpha = (
            pynutil.insert("decimal { ")
            + decimal.final_graph_wo_negative
            + pynini.cross('-', '')
            + pynutil.insert(" } units: \"")
            + pynini.closure(NEMO_ALPHA, 1)
            + pynutil.insert("\"")
        )

        alpha_dash_decimal = (
            pynutil.insert("units: \"")
            + pynini.closure(NEMO_ALPHA, 1)
            + pynini.cross('-', '')
            + pynutil.insert("\"")
            + pynutil.insert(" decimal { ")
            + decimal.final_graph_wo_negative
            + pynutil.insert(" } preserve_order: true")
        )

        subgraph_fraction = (
            pynutil.insert("fraction { ") + fraction.graph + delete_space + pynutil.insert(" } ") + unit_plural
        )

        address = self.get_address_graph(cardinal)
        address = (
            pynutil.insert("units: \"address\" cardinal { integer: \"")
            + address
            + pynutil.insert("\" } preserve_order: true")
        )

        final_graph = (
            subgraph_decimal
            | subgraph_cardinal
            | cardinal_dash_alpha
            | alpha_dash_cardinal
            | decimal_dash_alpha
            | alpha_dash_decimal
            | subgraph_fraction
            | address
        )
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()

    def get_address_graph(self, cardinal):
        """
        Finite state transducer for classifying serial.
            The serial is a combination of digits, letters and dashes, e.g.:
            2788 San Tomas Expy, Santa Clara, CA 95051 ->
                units: "address" cardinal
                { integer: "two seven eight eight San Tomas Expressway Santa Clara California nine five zero five one" }
                 preserve_order: true
        """
        ordinal_verbalizer = OrdinalVerbalizer().graph
        ordinal_tagger = OrdinalTagger(cardinal=cardinal).graph
        ordinal_num = pynini.compose(
            pynutil.insert("integer: \"") + ordinal_tagger + pynutil.insert("\""), ordinal_verbalizer
        )

        address_num = pynini.closure(NEMO_DIGIT, 1) @ cardinal.single_digits_graph

        direction = (
            pynini.cross("E", "East")
            | pynini.cross("S", "South")
            | pynini.cross("W", "West")
            | pynini.cross("N", "North")
        )
        direction = pynini.closure(pynutil.add_weight(pynini.accep(NEMO_SPACE) + direction, -1), 0, 1)

        address_words = pynini.string_file(get_abs_path("data/address/address_words.tsv"))
        address_words = (
            pynini.accep(NEMO_SPACE)
            + pynini.closure(ordinal_num, 0, 1)
            + pynini.closure(NEMO_ALPHA | NEMO_SPACE, 1)
            + address_words
        )

        city = pynini.closure(NEMO_ALPHA | pynini.accep(NEMO_SPACE), 1)
        city = pynini.closure(pynini.cross(",", "") + pynini.accep(NEMO_SPACE) + city, 0, 1)

        state = pynini.invert(pynini.string_file(get_abs_path("data/address/states.tsv")))
        state = pynini.closure(pynini.cross(",", "") + pynini.accep(NEMO_SPACE) + state, 0, 1)

        zip_code = pynini.compose(NEMO_DIGIT ** 5, cardinal.single_digits_graph)
        zip_code = pynini.closure(
            pynutil.add_weight(
                pynini.closure(pynini.cross(",", ""), 0, 1) + pynini.accep(NEMO_SPACE) + zip_code, -100
            ),
            0,
            1,
        )

        address = (
            address_num
            + direction
            + address_words
            + pynini.closure(pynini.cross(".", ""), 0, 1)
            + city
            + state
            + zip_code
        )
        return address
