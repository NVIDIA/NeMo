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
from nemo_text_processing.text_normalization.graph_utils import NEMO_DIGIT, NEMO_SIGMA, GraphFst
from nemo_text_processing.text_normalization.ru.taggers.number_names import NumberNamesFst
from nemo_text_processing.text_normalization.taggers.date import get_hundreds_graph

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

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        if deterministic:
            raise ValueError('Ru TN only support non-deterministic cases and produces multiple normalization options.')

        n = NumberNamesFst()
        cardinal = n.cardinal_number_names
        ordinal = n.ordinal_number_names

        t = pynini.Far(get_abs_path('ru/data/utils/universal_thousands_punct.far'))
        b = pynini.Far(get_abs_path('ru/data/utils/util_byte.far'), mode='r')

        # TODO use NEMO_SIGMA?
        sigma_star = pynini.closure(b['kBytes'])
        nominatives = pynini.string_file(get_abs_path("ru/data/nominatives.tsv"))
        nominative_filter = pynutil.add_weight(pynini.cross("", ""), -1)
        nominative_filter = nominatives @ pynini.cdrewrite(
            nominative_filter, pynini.union("[BOS]", " "), pynini.union(" ", "[EOS]"), sigma_star
        )

        self.graph = cardinal
        # skipped I and D in numbers.grm

        # graph = pynini.Far(
        #     '/home/ebakhturina/itn_cg/TextNormalizationCoveringGrammars/src/ru/verbalizer/numbers.far', mode='r'
        # )['CARDINAL_DEFAULT']
        #
        # graph = graph.invert().optimize()
        #
        # graph = pynutil.insert("value: \"") + graph + pynutil.insert("\"")
        self.graph = self.add_tokens(self.graph)
        self.fst = self.graph

        # Since we know this is the default for Russian, it's fair game to set it.
        separators = t['dot_thousands'] | t['no_delimiter']
