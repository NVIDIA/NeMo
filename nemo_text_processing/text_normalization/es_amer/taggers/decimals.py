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
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.es_amer.graph_utils import (
    cardinal_separator,
    decimal_separator,
    strip_cardinal_apocope,
)
from nemo_text_processing.text_normalization.es.utils import get_abs_path

try:
    import pynini
    from pynini.lib import pynutil

    quantities = pynini.string_file(get_abs_path("data/numbers/quantities.tsv"))

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False
    quantities = None


def get_quantity(decimal: 'pynini.FstLike', cardinal_graph: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Returns FST that transforms either a cardinal or decimal followed by a quantity into a numeral,
    e.g. 2 million -> integer_part: "dos" quantity: "millones"
    e.g. 2.4 million -> integer_part: "dos" fractional_part: "quatro" quantity: "million"

    Args:
        decimal: decimal FST
        cardinal_up_to_hundred: cardinal FST
    """
    numbers = pynini.closure(NEMO_DIGIT, 1, 6) @ cardinal_graph
    numbers = pynini.cdrewrite(pynutil.delete(cardinal_separator), "", "", NEMO_SIGMA) @ numbers

    res = (
        pynutil.insert("integer_part: \"")
        + numbers  # The cardinal we're passing only produces 'un' for one, so gender agreement is safe (all quantities are masculine). LImit to 10^6 power.
        + pynutil.insert("\"")
        + NEMO_SPACE
        + pynutil.insert("quantity: \"")
        + quantities
        + pynutil.insert("\"")
    )
    res |= decimal + NEMO_SPACE + pynutil.insert("quantity: \"") + quantities + pynutil.insert("\"")
    return res


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal, e.g.
        -11,4006 billion -> decimal { negative: "true" integer_part: "once"  fractional_part: "" quantity: "billion" preserve_order: true }
        1 billion -> decimal { integer_part: "un" quantity: "billón" preserve_order: true }
    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        graph_digit = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).invert()
        graph_zero = pynini.string_file(get_abs_path("data/numbers/zero.tsv")).invert()
        graph_digit |= graph_zero

        if not deterministic:
            graph = pynini.union(graph_digit, cardinal.hundreds, cardinal.tens)
            graph = graph + pynini.closure(insert_space + graph)

        else:
            # General pattern seems to be 1-3 digits: map as cardinal. \
            graph = cardinal.hundreds | graph_digit | cardinal.tens  # Read up to group of three
            graph |= (
                graph_zero + insert_space + pynini.closure(graph_digit + insert_space) + graph_digit
            )  # String of digits after 0, or just string of zeros
            # Default to string of digits after
            graph |= graph_digit + pynini.closure(insert_space + graph_digit, 3)

        # Need to strip apocope everywhere BUT end of string
        reverse_apocope = pynini.string_map([("un", "uno"), ("ún", "uno")])
        apply_reverse_apocope = pynini.cdrewrite(reverse_apocope, "", NEMO_SPACE, NEMO_SIGMA)
        graph @= apply_reverse_apocope

        # Technically SI standards argues for writing decimals grouped by three, e.g. (1,333 333). This removes any possible spaces
        strip_formatting = pynini.cdrewrite(delete_space, "", "", NEMO_SIGMA)
        graph = strip_formatting @ graph

        self.graph = graph.optimize()

        graph_separator = pynutil.delete(decimal_separator)
        optional_graph_negative = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)

        self.graph_fractional = pynutil.insert("fractional_part: \"") + self.graph + pynutil.insert("\"")

        # Integer graph maintains apocope except for ones place
        graph_integer = (
            strip_cardinal_apocope(cardinal.graph)
            if deterministic
            else pynini.union(cardinal.graph, strip_cardinal_apocope(cardinal.graph))
        )  # Gives us forms w/ and w/o apocope
        self.graph_integer = pynutil.insert("integer_part: \"") + graph_integer + pynutil.insert("\"")
        final_graph_wo_sign = self.graph_integer + graph_separator + insert_space + self.graph_fractional

        self.final_graph_wo_negative = (
            final_graph_wo_sign | get_quantity(final_graph_wo_sign, cardinal.graph).optimize()
        )
        final_graph = optional_graph_negative + self.final_graph_wo_negative

        final_graph += pynutil.insert(" preserve_order: true")
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()
