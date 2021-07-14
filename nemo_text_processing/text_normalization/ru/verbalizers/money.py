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

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst
from nemo_text_processing.text_normalization.ru.alphabet import RU_ALPHA

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):

    PYNINI_AVAILABLE = False


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money {  integer_part: "пять" currency: "рублей" } -> пять рублей

    Args:
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        # integer = (
        #     pynutil.delete("integer_part:")
        #     + delete_space
        #     + pynutil.delete("\"")
        #     + pynini.closure(NEMO_NOT_QUOTE, 1)
        #     + pynutil.delete("\"")
        # )
        # unit = (
        #     pynutil.delete("currency:")
        #     + delete_space
        #     + pynutil.delete("\"")
        #     + pynini.closure(NEMO_NOT_QUOTE, 1)
        #     + pynutil.delete("\"")
        # )
        # graph = integer + delete_space + pynutil.insert(" ") + unit
        #
        # delete_tokens = self.delete_tokens(graph)
        # self.fst = delete_tokens.optimize()

        graph = pynini.closure(RU_ALPHA | " ")
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
