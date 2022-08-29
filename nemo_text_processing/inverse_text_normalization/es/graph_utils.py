# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.es.utils import get_abs_path
from pynini.lib import pynutil


def int_to_roman(fst: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Alters given fst to convert Arabic numerals into Roman integers (lower cased). Valid for values up to 3999.
    e.g.
        "5" -> "v"
        "treinta y uno" -> "xxxi"
    Args:
        fst: Any fst. Composes fst onto Roman conversion outputs.
    """

    def _load_roman(file: str):
        roman_numerals = pynini.string_file(get_abs_path(file))
        return pynini.invert(roman_numerals)

    digit = _load_roman("data/roman/digit.tsv")
    ties = _load_roman("data/roman/ties.tsv")
    hundreds = _load_roman("data/roman/hundreds.tsv")
    thousands = _load_roman("data/roman/thousands.tsv")

    graph = (
        digit
        | ties + (digit | pynutil.add_weight(pynutil.delete("0"), 0.01))
        | (
            hundreds
            + (ties | pynutil.add_weight(pynutil.delete("0"), 0.01))
            + (digit | pynutil.add_weight(pynutil.delete("0"), 0.01))
        )
        | (
            thousands
            + (hundreds | pynutil.add_weight(pynutil.delete("0"), 0.01))
            + (ties | pynutil.add_weight(pynutil.delete("0"), 0.01))
            + (digit | pynutil.add_weight(pynutil.delete("0"), 0.01))
        )
    ).optimize()

    return fst @ graph
