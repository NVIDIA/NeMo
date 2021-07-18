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
from nemo_text_processing.text_normalization.en.utils import load_labels
from nemo_text_processing.text_normalization.ru.utils import get_abs_path

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class AlternativeFormatsFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        -23 -> cardinal { negative: "true"  integer: "twenty three" } }
    """

    def __init__(self):
        one_alternatives = load_labels(get_abs_path('data/cardinals_alternatives.tsv'))
        one_thousand_map = []
        for k in one_alternatives:
            default, alternative = k
            one_thousand_map.append((alternative.split()[1], alternative))
        one_thousand_map = pynini.string_map(one_thousand_map)

        self.one_thousand_alternative = pynini.cdrewrite(one_thousand_map, "[BOS]", "", NEMO_SIGMA)

        t = pynini.Far(get_abs_path('data/utils/universal_thousands_punct.far'))
        self.separators = (
            pynutil.add_weight(t['dot_thousands'], 1)
            | pynutil.add_weight(t['no_delimiter'], 1)
            | pynutil.add_weight(t['space_thousands'], -1)
        )
