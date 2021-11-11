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

from nemo_text_processing.text_normalization.de.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    NEMO_WHITE_SPACE,
    SINGULAR_TO_PLURAL,
    GraphFst,
    delete_space,
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
        money {  "пять рублей" } -> пять рублей

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        keep_space = pynini.accep(" ")

        maj = pynutil.delete("currency_maj: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        min = pynutil.delete("currency_min: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        preserve_order = pynini.closure(
            pynutil.delete(" preserve_order: true")
            | (pynutil.delete(" field_order: \"") + NEMO_NOT_QUOTE + pynutil.delete("\"")),
            1,
        )

        fractional_part = (
            pynutil.delete("fractional_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        )

        integer_part = pynutil.delete("integer_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        optional_add_and = pynini.closure(pynutil.insert(" und "), 0, 1)

        #  *** currency_maj
        graph_integer = integer_part + keep_space + maj

        #  *** currency_maj + (***) | ((und) *** current_min)
        graph_integer_with_minor = (
            integer_part
            + keep_space
            + maj
            + keep_space
            + (fractional_part | (optional_add_and + fractional_part + keep_space + min))
            + preserve_order
        )

        # *** komma *** currency_maj
        graph_decimal = integer_part + keep_space + pynutil.insert("komma ") + fractional_part + keep_space + maj

        # *** current_min
        graph_minor = fractional_part + keep_space + min + preserve_order

        graph = graph_integer | graph_integer_with_minor | graph_decimal | graph_minor

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
