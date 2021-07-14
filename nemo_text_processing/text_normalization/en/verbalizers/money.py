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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    get_abs_path,
    insert_space,
)

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):

    PYNINI_AVAILABLE = False


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { integer_part: "twelve" fractional_part: "o five" currency: "dollars" } -> twelve o five dollars

    Args:
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        def _get_minor_currencies(file):
            minor_currencies = []
            with open(get_abs_path(file), 'r') as f:
                for line in f:
                    min_cur = line.strip()
                    minor_currencies.append(pynutil.insert(min_cur))
            return minor_currencies

        unit = (
            pynutil.delete("currency:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        graph = decimal.numbers + delete_space + pynutil.insert(" ") + unit

        if not deterministic:
            minor_currencies_singular = _get_minor_currencies("data/currency/currency_minor_one.tsv")
            minor_currencies_singular = pynini.union(*minor_currencies_singular)
            minor_currencies_singular = (
                pynini.closure(NEMO_NOT_QUOTE)
                + (
                    pynini.accep("one")
                    | pynini.cross("zero one", "one")
                    | pynini.cross("oh one", "one")
                    | pynini.cross(" o one", " one")
                )
                + insert_space
                + minor_currencies_singular
            )

            minor_currencies_plural = _get_minor_currencies("data/currency/currency_minor.tsv")
            minor_currencies_plural = insert_space + pynini.union(*minor_currencies_plural)

            fractional_default = (
                pynutil.delete("fractional_part:")
                + delete_space
                + pynutil.delete("\"")
                + ((pynini.closure(NEMO_NOT_QUOTE, 1) + minor_currencies_plural) | minor_currencies_singular)
                + pynutil.delete("\"")
            )

            # $2.00 {two zero zero dollars} -> two dollars
            fractional_with_zeros = (
                pynutil.delete("fractional_part:")
                + delete_space
                + pynutil.delete("\"")
                + pynini.cross('zero', '')
                + pynini.closure(pynini.cross(' zero', ''))
                + delete_space
                + pynutil.delete("\"")
                + delete_space
            )

            fractional = fractional_with_zeros | fractional_default

            graph |= (
                decimal.integer
                + delete_space
                + insert_space
                + unit
                + delete_space
                + insert_space
                + pynini.closure(pynutil.insert("and "), 0, 1)
                + fractional
            )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
