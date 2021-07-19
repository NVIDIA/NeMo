# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

# Adapted from https://github.com/google/TextNormalizationCoveringGrammars
# Russian minimally supervised number grammar.
#
# Supports cardinals and ordinals in all inflected forms.
#
# The language-specific acceptor G was compiled with digit, teen, decade,
# century, and big power-of-ten preterminals. The lexicon transducer is
# highly ambiguous, but no LM is used.

# Intersects the universal factorization transducer (F) with language-specific
# acceptor (G).


from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, GraphFst
from nemo_text_processing.text_normalization.ru.utils import get_abs_path

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

    def __init__(self, number_names: GraphFst, alternative_formats: GraphFst, deterministic=False):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        one_thousand_alternative = alternative_formats.one_thousand_alternative
        separators = alternative_formats.separators

        ordinal = number_names.ordinal_number_names

        ordinal |= ordinal @ one_thousand_alternative
        ordinal_numbers = separators @ ordinal

        # to handle cases like 2-ая
        endings = pynini.string_file(get_abs_path("data/ordinal_endings.tsv"))
        not_dash = pynini.closure(pynini.difference(NEMO_SIGMA, "-"))
        del_ending = pynini.cdrewrite(pynini.cross("-" + not_dash, ""), "", "[EOS]", NEMO_SIGMA)
        ordinal_numbers_marked = (
            ((separators @ ordinal).optimize() + pynini.accep("-") + not_dash).optimize()
            @ (NEMO_SIGMA + endings).optimize()
            @ del_ending
        ).optimize()

        self.ordinal_numbers = ordinal_numbers
        # "03" -> remove leading zeros and verbalize
        leading_zeros = pynini.closure(pynini.cross("0", ""))
        self.ordinal_numbers_with_leading_zeros = (leading_zeros + ordinal_numbers).optimize()

        final_graph = (ordinal_numbers | ordinal_numbers_marked).optimize()
        final_graph = pynutil.insert("integer: \"") + final_graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()


if __name__ == '__main__':
    fst = OrdinalFst(deterministic=False).ordinal_numbers
    from pynini.lib import rewrite

    pred = rewrite.rewrites("2", fst)
    assert len(pred) > 1 and "вторая" in pred
    assert rewrite.rewrites("2-ая", fst) == ['вторая']
    print(rewrite.rewrites("2-ая", fst))
