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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, insert_space
from nemo_text_processing.text_normalization.en.verbalizers.ordinal import OrdinalFst

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fraction
        e.g. tokens { fraction { integer: "twenty three" numerator: "four" denominator: "five" } } ->
        twenty three four fifth

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)
        suffix = OrdinalFst().suffix

        integer = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\" ")
        numerator = pynutil.delete("numerator: \"") + pynini.closure(NEMO_NOT_QUOTE) + pynutil.delete("\" ")
        numerator_one = pynutil.delete("numerator: \"") + pynini.accep("one") + pynutil.delete("\" ")
        denominator = pynutil.delete("denominator: \"") + (
            pynini.closure(NEMO_NOT_QUOTE) @ suffix | pynutil.add_weight(pynini.cross('four', 'quarter'), -1)
        )
        conjunction = pynutil.insert("and ")
        if not deterministic:
            conjunction = pynini.closure(conjunction, 0, 1)

        integer = pynini.closure(integer + insert_space + conjunction, 0, 1)

        denominator_half = pynini.cross("numerator: \"one\" denominator: \"two\"", "a half")
        denominator_one_two = pynini.cross("denominator: \"one\"", "over one") | pynini.cross(
            "denominator: \"two\"", "halves"
        )
        fraction_default = pynutil.add_weight(
            numerator + insert_space + denominator + pynutil.insert("s") + pynutil.delete("\""), 0.001
        )
        fraction_with_one = pynutil.add_weight(
            numerator_one + insert_space + denominator + pynutil.delete("\""), 0.0001
        )

        graph = integer + denominator_half | (fraction_with_one | fraction_default)
        graph |= pynutil.add_weight(pynini.cross("numerator: \"one\" denominator: \"two\"", "one half"), -1)
        graph |= (numerator | numerator_one) + insert_space + denominator_one_two

        self.graph = graph
        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()
