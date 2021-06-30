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


from nemo_text_processing.text_normalization.data_loader_utils import get_abs_path
from nemo_text_processing.text_normalization.graph_utils import NEMO_SIGMA, GraphFst
from nemo_text_processing.text_normalization.ru.taggers.number_names import NumberNamesFst
from nemo_text_processing.text_normalization.ru.taggers.numbers_alternatives import AlternativeFormatsFst

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g. 
        23 -> ordinal { integer: "twenty third" } }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic=False):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        print('Ru TN only support non-deterministic cases and produces multiple normalization options.')

        alternative_formats = AlternativeFormatsFst()
        one_thousand_alternative = alternative_formats.one_thousand_alternative
        separators = alternative_formats.separators

        n = NumberNamesFst()
        ordinal = n.ordinal_number_names

        ordinal |= ordinal @ one_thousand_alternative
        ordinal_numbers = separators @ ordinal

        # to handle cases like 2-ая
        endings = pynini.string_file(get_abs_path("ru/data/ordinal_endings.tsv"))
        not_dash = pynini.closure(pynini.difference(NEMO_SIGMA, "-"))
        del_ending = pynini.cdrewrite(pynini.cross("-" + not_dash, ""), "", "[EOS]", NEMO_SIGMA)
        ordinal_numbers_marked = (
            ((separators @ ordinal).optimize() + pynini.accep("-") + not_dash).optimize()
            @ (NEMO_SIGMA + endings).optimize()
            @ del_ending
        ).optimize()

        self.ordinal_numbers = (ordinal_numbers | ordinal_numbers_marked).optimize()

        final_graph = pynutil.insert("integer: \"") + self.ordinal_numbers + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()


if __name__ == '__main__':
    fst = OrdinalFst(deterministic=False).ordinal_numbers
    from pynini.lib import rewrite

    pred = rewrite.rewrites("2", fst)
    assert len(pred) > 1 and "вторая" in pred
    assert rewrite.rewrites("2-ая", fst) == ['вторая']
    print(rewrite.rewrites("2-ая", fst))
