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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NON_BREAKING_SPACE,
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.ru.alphabet import RU_ALPHA
from nemo_text_processing.text_normalization.ru.utils import get_abs_path

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure,  e.g.
        "2 кг" -> measure { cardinal { integer: "два килограма" } }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        # adding weight to make sure the space is preserved for ITN
        delete_space = pynini.closure(
            pynutil.add_weight(pynutil.delete(pynini.union(NEMO_SPACE, NEMO_NON_BREAKING_SPACE)), -1), 0, 1
        )
        cardinal_graph = cardinal.cardinal_numbers

        graph_unit = pynini.string_file(get_abs_path("data/measurements.tsv"))

        graph_unit_plural = graph_unit
        optional_graph_negative = cardinal.optional_graph_negative

        graph_unit2 = pynini.cross("/", "в") + delete_space + pynutil.insert(NEMO_NON_BREAKING_SPACE) + graph_unit

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

        subgraph_decimal = decimal.final_graph + delete_space + unit_plural

        cardinal_space_plural = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + ((NEMO_SIGMA - "1") @ cardinal_graph)
            + delete_space
            + pynutil.insert("\"")
            + pynutil.insert(" } ")
            + unit_plural
        )

        one = pynini.compose(pynini.accep("1"), cardinal_graph).optimize()
        cardinal_space_singular = (
            pynutil.insert("cardinal { ")
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + one
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
            + pynini.closure(RU_ALPHA, 1)
            + pynutil.insert("\"")
        )

        alpha_dash_cardinal = (
            pynutil.insert("units: \"")
            + pynini.closure(RU_ALPHA, 1)
            + pynini.cross('-', '')
            + pynutil.insert("\"")
            + pynutil.insert(" cardinal { integer: \"")
            + cardinal_graph
            + pynutil.insert("\" } preserve_order: true")
        )

        decimal_dash_alpha = (
            decimal.final_graph
            + pynini.cross('-', '')
            + pynutil.insert(" units: \"")
            + pynini.closure(RU_ALPHA, 1)
            + pynutil.insert("\"")
        )

        alpha_dash_decimal = (
            pynutil.insert("units: \"")
            + pynini.closure(RU_ALPHA, 1)
            + pynini.cross('-', '')
            + pynutil.insert("\"")
            + pynutil.insert(" decimal { ")
            + decimal.final_graph
            + pynutil.insert(" } preserve_order: true")
        )

        self.tagger_graph_default = (subgraph_decimal | cardinal_space_singular | cardinal_space_plural).optimize()

        tagger_graph = (
            self.tagger_graph_default
            | cardinal_dash_alpha
            | alpha_dash_cardinal
            | decimal_dash_alpha
            | alpha_dash_decimal
        ).optimize()

        # verbalizer
        unit = pynutil.delete("units: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"") + delete_space

        optional_sign = pynini.closure(pynini.cross("negative: \"true\" ", "минус "), 0, 1)
        integer = pynutil.delete(" \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        integer_part = pynutil.delete("integer_part:") + integer
        fractional_part = pynutil.delete("fractional_part:") + integer
        graph_decimal = optional_sign + integer_part + pynini.accep(" ") + fractional_part

        graph_decimal = pynutil.delete("decimal {") + delete_space + graph_decimal + delete_space + pynutil.delete("}")

        graph_cardinal = (
            pynutil.delete("cardinal {")
            + delete_space
            + optional_sign
            + pynutil.delete("integer: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + delete_space
            + pynutil.delete("}")
        )

        verbalizer_graph = (graph_cardinal | graph_decimal) + delete_space + insert_space + unit

        # SH adds "preserve_order: true" by default
        preserve_order = pynutil.delete("preserve_order:") + delete_space + pynutil.delete("true") + delete_space
        verbalizer_graph |= (
            unit + insert_space + (graph_cardinal | graph_decimal) + delete_space + pynini.closure(preserve_order)
        )
        self.verbalizer_graph = verbalizer_graph.optimize()

        final_graph = (tagger_graph @ verbalizer_graph).optimize()
        self.fst = self.add_tokens(final_graph).optimize()
