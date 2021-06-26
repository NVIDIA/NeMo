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

from nemo_text_processing.text_normalization.data_loader_utils import get_abs_path
from nemo_text_processing.text_normalization.graph_utils import GraphFst

try:
    import pynini
    from pynini.lib import pynutil, rewrite

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class NumberNamesFst(GraphFst):
    """
    # TODO
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="number_names", kind="classify", deterministic=deterministic)

        a = pynini.Far(get_abs_path('ru/data/utils/util_arithmetic.far'), mode='r')
        d = a['DELTA_STAR']
        f = a['IARITHMETIC_RESTRICTED']
        g = pynini.Fst.read(get_abs_path('ru/data/utils/g.fst'))
        fg = (d @ (f @ (f @ (f @ g).optimize()).optimize()).optimize()).optimize()

        assert rewrite.top_rewrite("230", fg) == "(+ 200 30 +)"

        # Compiles lexicon transducers (L).
        cardinal_name = pynini.string_file(get_abs_path("ru/data/cardinals.tsv"))  # .optimize()
        cardinal_l = (pynini.closure(cardinal_name + pynini.accep(" ")) + cardinal_name).optimize()

        # TODO fix e issues in ordinal.tsv vocabulary
        ordinal_name = pynini.string_file(get_abs_path("ru/data/ordinals.tsv"))  # , input_token_type='utf-8')
        ordinal_l = (pynini.closure(cardinal_name + pynini.accep(" ")) + ordinal_name).optimize()

        # Composes L with the leaf transducer (P), then composes that with FG.
        p = a['LEAVES']
        self.ordinal_number_names = (fg @ (p @ ordinal_l)).optimize()
        self.cardinal_number_names = (fg @ (p @ cardinal_l)).optimize()

        examples = {"tn": ["2", "53"], "itn": ["два", "пятьдесят три"]}
        invert = pynini.invert(self.cardinal_number_names)
        invert = invert.optimize()
        for written, spoken in zip(examples['tn'], examples['itn']):
            try:
                assert spoken in rewrite.rewrites(written, self.cardinal_number_names), 'TN failed'
                assert written == rewrite.top_rewrite(spoken, invert), 'ITN failed'
            except:
                import pdb

                pdb.set_trace()
                print()


if __name__ == "__main__":
    NumberNamesFst()
