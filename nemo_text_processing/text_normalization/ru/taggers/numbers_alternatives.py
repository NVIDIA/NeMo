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


from nemo_text_processing.text_normalization.data_loader_utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.graph_utils import NEMO_SIGMA, GraphFst

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
        one_alternatives = load_labels(get_abs_path('ru/data/cardinals_alternatives.tsv'))
        one_thousand_map = []
        for k in one_alternatives:
            default, alternative = k
            one_thousand_map.append((alternative.split()[1], alternative))
        one_thousand_map = pynini.string_map(one_thousand_map)

        self.one_thousand_alternative = pynini.cdrewrite(one_thousand_map, "[BOS]", "", NEMO_SIGMA)

        t = pynini.Far(get_abs_path('ru/data/utils/universal_thousands_punct.far'))
        b = pynini.Far(get_abs_path('ru/data/utils/util_byte.far'), mode='r')

        # TODO use NEMO_SIGMA?
        # TODO nominatives - what's their purpose here?
        sigma_star = pynini.closure(b['kBytes'])
        nominatives = pynini.string_file(get_abs_path("ru/data/nominatives.tsv"))
        nominative_filter = pynutil.add_weight(pynini.cross("", ""), -1)
        nominative_filter = nominatives @ pynini.cdrewrite(
            nominative_filter, pynini.union("[BOS]", " "), pynini.union(" ", "[EOS]"), sigma_star
        )

        # skipped I and D in numbers.grm

        # TODO add support for space separated numbers "12 000"
        self.separators = t['dot_thousands'] | t['no_delimiter'] | t['space_thousands']
