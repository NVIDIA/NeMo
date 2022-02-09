# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import pynini
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, NEMO_SPACE
from nemo_text_processing.text_normalization.es.utils import get_abs_path, load_labels
from pynini.lib import pynutil

cardinal_separator = pynini.string_map([".", NEMO_SPACE])
decimal_separator = pynini.accep(",")
ones = pynini.union("un", "ún")
accents = pynini.string_map([("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u")])

digits = pynini.string_file(get_abs_path("data/numbers/digit.tsv")).project("input")
tens = pynini.string_file(get_abs_path("data/numbers/ties.tsv")).project("input")
teens = pynini.string_file(get_abs_path("data/numbers/teen.tsv")).project("input")
twenties = pynini.string_file(get_abs_path("data/numbers/twenties.tsv")).project("input")
hundreds = pynini.string_file(get_abs_path("data/numbers/hundreds.tsv")).project("input")

one_to_one_hundred = pynini.union(digits, tens, teens, twenties, tens + pynini.accep(" y ") + digits)


def strip_accent(fst):
    return fst @ pynini.cdrewrite(accents, "", "", NEMO_SIGMA)


def shift_cardinal_gender(fst):
    # Converts number string to feminine gender for cardinal string (i.e. only affects place values below one million - higher powers of ten are technically masculine nouns)
    fem_hundreds = hundreds @ pynini.cdrewrite(pynini.cross("ientos", "ientas"), "", "", NEMO_SIGMA)
    fem_ones = pynini.cross("un", "una") | pynini.cross("ún", "una") | pynini.cross("uno", "una")

    before_mil = (
        NEMO_SPACE
        + (pynini.accep("mil") | pynini.accep("milésimo"))
        + pynini.closure(NEMO_SPACE + hundreds, 0, 1)
        + pynini.closure(NEMO_SPACE + one_to_one_hundred, 0, 1)
        + pynini.union(pynini.accep("[EOS]"), pynini.accep("\""), decimal_separator)
    )
    before_double_digits = pynini.closure(NEMO_SPACE + one_to_one_hundred, 0, 1) + pynini.union(
        pynini.accep("[EOS]"), pynini.accep("\"")
    )

    fem_allign = pynini.cdrewrite(fem_hundreds, "", before_mil, NEMO_SIGMA)  # doscientas mil dosciento
    fem_allign @= pynini.cdrewrite(fem_hundreds, "", before_double_digits, NEMO_SIGMA)  # doscientas mil doscienta
    fem_allign @= pynini.cdrewrite(
        fem_ones, "", pynini.union("[EOS]", "\"", decimal_separator), NEMO_SIGMA
    )  # If before a quote or EOS, we know it's the end of a string

    return fst @ fem_allign


def shift_number_gender(fst):
    # Shifts gender without boundary concerns (for ordinals and fractions when the higher powers actually acquire gender)
    fem_hundreds = pynini.cross("ientos", "ientas")
    fem_ones = pynini.cross("un", "una") | pynini.cross("ún", "una") | pynini.cross("uno", "una")
    fem_allign = pynini.cdrewrite(fem_hundreds, "", "", NEMO_SIGMA)
    fem_allign @= pynini.cdrewrite(
        fem_ones, "", pynini.union(NEMO_SPACE, pynini.accep("[EOS]"), pynini.accep("\"")), NEMO_SIGMA
    )  # If before a quote or EOS, we know it's the end of a string

    return fst @ fem_allign


def strip_cardinal_apocope(
    fst,
):  # Since cardinals use apocope by default, this only needs to act on the last instance of one
    strip = pynini.cross("un", "uno") | pynini.cross("ún", "uno")
    strip = pynini.cdrewrite(strip, "", pynini.union("[EOS]", "\""), NEMO_SIGMA)
    return fst @ strip


def roman_to_int(fst):
    def _load_roman(file: str):
        roman = load_labels(get_abs_path(file))
        roman_numerals = [(x, y) for x, y in roman] + [(x.upper(), y) for x, y in roman]
        return pynini.string_map(roman_numerals)

    digit = _load_roman("data/roman/digit.tsv")
    ties = _load_roman("data/roman/ties.tsv")
    hundreds = _load_roman("data/roman/hundreds.tsv")

    graph = (
        digit
        | ties + (digit | pynutil.add_weight(pynutil.insert("0"), 0.01))
        | (
            hundreds
            + (ties | pynutil.add_weight(pynutil.insert("0"), 0.01))
            + (digit | pynutil.add_weight(pynutil.insert("0"), 0.01))
        )
    ).optimize()

    return graph @ fst
