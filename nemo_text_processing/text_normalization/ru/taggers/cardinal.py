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

from nemo_text_processing.text_normalization.graph_utils import GraphFst, insert_space
from nemo_text_processing.text_normalization.ru.taggers.number_names import NumberNamesFst
from nemo_text_processing.text_normalization.ru.taggers.numbers_alternatives import AlternativeFormatsFst

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g. 
        -23 -> cardinal { negative: "true"  integer: "twenty three" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = False):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        print('Ru TN only support non-deterministic cases and produces multiple normalization options.')
        n = NumberNamesFst()
        cardinal = n.cardinal_number_names

        alternative_formats = AlternativeFormatsFst()
        one_thousand_alternative = alternative_formats.one_thousand_alternative
        separators = alternative_formats.separators

        cardinal |= cardinal @ one_thousand_alternative
        cardinal_numbers = separators @ cardinal
        self.optional_graph_negative = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", "\"true\"") + insert_space, 0, 1
        )
        self.cardinal_numbers = cardinal_numbers
        self.cardinal_numbers_with_optional_negative = (
            self.optional_graph_negative + pynutil.insert("integer: \"") + cardinal_numbers + pynutil.insert("\"")
        )

        # "03" -> remove leading zeros and verbalize
        leading_zeros = pynini.closure(pynini.cross("0", ""))
        self.cardinal_numbers_with_leading_zeros = (leading_zeros + cardinal_numbers).optimize()
        final_graph = (
            self.optional_graph_negative
            + pynutil.insert("integer: \"")
            + self.cardinal_numbers_with_leading_zeros
            + pynutil.insert("\"")
        ).optimize()

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()


if __name__ == '__main__':
    fst = CardinalFst()
    from pynini.lib.rewrite import rewrites

    d = {
        "147691": "сто сорок семь тысяч шестьсот девяносто один\"",
        "2300": "две тысячи триста\"",
        "-2300": "две тысячи триста\"",
        "002300": "две тысячи триста\"",
        "2.300": "две тысячи триста\"",
    }

    def _test(written, spoken):
        output = rewrites(written, fst.fst)
        if written == '147691':
            import pdb

            pdb.set_trace()
        test_passed = False
        for x in output:
            if spoken in x:
                test_passed = True

        assert test_passed, f'{written} failed'

    for k, v in d.items():
        _test(k, v)
