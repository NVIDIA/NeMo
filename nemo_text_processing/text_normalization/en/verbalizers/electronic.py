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
    NEMO_ALPHA,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    insert_space,
)
from nemo_text_processing.text_normalization.en.utils import get_abs_path

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class ElectronicFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic
        e.g. tokens { electronic { username: "cdf1" domain: "abc.edu" } } -> c d f one at a b c dot e d u

    Args:
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="verbalize", deterministic=deterministic)
        graph_digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv"))).optimize()
        graph_symbols = pynini.string_file(get_abs_path("data/electronic/symbols.tsv")).optimize()
        user_name = (
            pynutil.delete("username:")
            + delete_space
            + pynutil.delete("\"")
            + (
                pynini.closure(
                    pynutil.add_weight(graph_digit + insert_space, 1.09)
                    | pynutil.add_weight(pynini.closure(graph_symbols + pynutil.insert(" ")), 1.09)
                    | pynutil.add_weight(NEMO_NOT_QUOTE + insert_space, 1.1)
                )
            )
            + pynutil.delete("\"")
        )

        domain_default = (
            pynini.closure(NEMO_NOT_QUOTE + insert_space)
            + pynini.cross(".", "dot ")
            + NEMO_NOT_QUOTE
            + pynini.closure(insert_space + NEMO_NOT_QUOTE)
        )

        server_default = (
            pynini.closure((graph_digit | NEMO_ALPHA) + insert_space, 1)
            + pynini.closure(graph_symbols + insert_space)
            + pynini.closure((graph_digit | NEMO_ALPHA) + insert_space, 1)
        )
        server_common = pynini.string_file(get_abs_path("data/electronic/server_name.tsv")) + insert_space

        domain_common = pynini.cross(".", "dot ") + pynini.string_file(get_abs_path("data/electronic/domain.tsv"))

        domain = (
            pynutil.delete("domain:")
            + delete_space
            + pynutil.delete("\"")
            + (pynutil.add_weight(server_common, 1.09) | pynutil.add_weight(server_default, 1.1))
            + (pynutil.add_weight(domain_common, 1.09) | pynutil.add_weight(domain_default, 1.1))
            + delete_space
            + pynutil.delete("\"")
        )

        graph = (
            pynini.closure(user_name + delete_space + pynutil.insert("at ") + delete_space, 0, 1)
            + domain
            + delete_space
        )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
