# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright 2017 Google Inc.
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

from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA
from nemo_text_processing.text_normalization.ru.utils import get_abs_path, load_labels

try:
    import pynini
    from pynini.lib import pynutil
    from pynini.lib import rewrite

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


def get_number_names():
    """
    Creates numbers names.

    Based on: 1) Gorman, K., and Sproat, R. 2016. Minimally supervised number normalization.
    Transactions of the Association for Computational Linguistics 4: 507-519.
    and 2) Ng, A. H., Gorman, K., and Sproat, R. 2017.
    Minimally supervised written-to-spoken text normalization. In ASRU, pages 665-670.
    """
    a = pynini.Far(get_abs_path('data/utils/util_arithmetic.far'), mode='r')
    d = a['DELTA_STAR']
    f = a['IARITHMETIC_RESTRICTED']
    g = pynini.Fst.read(get_abs_path('data/utils/g.fst'))
    fg = (d @ (f @ (f @ (f @ g).optimize()).optimize()).optimize()).optimize()
    assert rewrite.top_rewrite("230", fg) == "(+ 200 30 +)"

    # Compiles lexicon transducers (L).
    cardinal_name_nominative = pynini.string_file(
        get_abs_path("data/numbers/1_cardinals_nominative_именительный.tsv")
    ).optimize()
    cardinal_name_genitive = pynini.string_file(
        get_abs_path("data/numbers/2_cardinals_genitive_родительный.tsv")
    ).optimize()
    cardinal_name_dative = pynini.string_file(get_abs_path("data/numbers/3_cardinals_dative_датильный.tsv")).optimize()
    cardinal_name_accusative = pynini.string_file(
        get_abs_path("data/numbers/4_cardinals_accusative_винительный.tsv")
    ).optimize()
    cardinal_name_instrumental = pynini.string_file(
        get_abs_path("data/numbers/5_cardinals_instrumental_творительный.tsv")
    ).optimize()
    cardinal_name_prepositional = pynini.string_file(
        get_abs_path("data/numbers/6_cardinals_prepositional_предложный.tsv")
    ).optimize()

    cardinal_l = (pynini.closure(cardinal_name_nominative + pynini.accep(" ")) + cardinal_name_nominative).optimize()
    for case in [
        cardinal_name_genitive,
        cardinal_name_dative,
        cardinal_name_accusative,
        cardinal_name_instrumental,
        cardinal_name_prepositional,
    ]:
        cardinal_l |= (pynini.closure(case + pynini.accep(" ")) + case).optimize()

    # Numbers up to 1000 in nominative case (to use, for example, with telephone)
    nominative_up_to_thousand_name = pynini.string_file(get_abs_path("data/numbers/cardinals_nominative_case.tsv"))
    nominative_up_to_thousand_name_l = (
        pynini.closure(nominative_up_to_thousand_name + pynini.accep(" ")) + nominative_up_to_thousand_name
    ).optimize()

    # (* 5 1000 *) -> 5000 handles complex ordinal numbers, e.g. "пятитысячный"
    complex_numbers = pynini.cross("(* 2 1000 *)", "2000")
    for number in range(3, 21):
        complex_numbers |= pynini.cross(f"(* {number} 1000 *)", f"{number}000")

    complex_numbers = (
        NEMO_SIGMA + pynutil.add_weight(complex_numbers, -1) + pynini.closure(pynini.union(" ", ")", "(", "+", "*"))
    )
    fg_ordinal = pynutil.add_weight(pynini.compose(fg, complex_numbers), -1) | fg
    ordinal_name = pynini.string_file(get_abs_path("data/numbers/ordinals.tsv"))
    ordinal_l = (pynini.closure(cardinal_name_nominative + pynini.accep(" ")) + ordinal_name).optimize()

    # Composes L with the leaf transducer (P), then composes that with FG.
    p = a['LEAVES']
    number_names = {}
    number_names['ordinal_number_names'] = (fg_ordinal @ (p @ ordinal_l)).optimize()
    number_names['cardinal_number_names'] = (fg @ (p @ cardinal_l)).optimize()
    number_names['nominative_up_to_thousand_names'] = (fg @ (p @ nominative_up_to_thousand_name_l)).optimize()
    return number_names


def get_alternative_formats():
    """
    Utils to get alternative formats for numbers.
    """
    one_alternatives = load_labels(get_abs_path('data/numbers/cardinals_alternatives.tsv'))
    one_thousand_map = []
    for k in one_alternatives:
        default, alternative = k
        one_thousand_map.append((alternative.split()[1], alternative))
    one_thousand_map = pynini.string_map(one_thousand_map)

    one_thousand_alternative = pynini.cdrewrite(one_thousand_map, "[BOS]", "", NEMO_SIGMA)

    t = pynini.Far(get_abs_path('data/utils/universal_thousands_punct.far'))
    separators = (
        pynutil.add_weight(t['dot_thousands'], 0.1)
        | pynutil.add_weight(t['no_delimiter'], -0.1)
        | pynutil.add_weight(t['space_thousands'], 0.1)
    )
    alternative_formats = {}
    alternative_formats['one_thousand_alternative'] = one_thousand_alternative
    alternative_formats['separators'] = separators
    return alternative_formats
