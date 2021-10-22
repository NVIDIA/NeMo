# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
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


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: email addresses
        e.g. "ab@nd.ru" -> electronic { username: "эй би собака эн ди точка ру" }

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="electronic", kind="classify", deterministic=deterministic)

        def get_input_symbols(f):
            accepted_symbols = []
            with open(f, 'r') as f:
                for line in f:
                    symbol, _ = line.split('\t')
                    accepted_symbols.append(pynini.accep(symbol))
            return accepted_symbols

        accepted_symbols = get_input_symbols(get_abs_path("data/electronic/symbols.tsv"))
        accepted_common_domains = get_input_symbols(get_abs_path("data/electronic/domain.tsv"))
        accepted_symbols = NEMO_ALPHA + pynini.closure(NEMO_ALPHA | NEMO_DIGIT | pynini.union(*accepted_symbols))
        graph_symbols = pynini.string_file(get_abs_path("data/electronic/symbols.tsv")).optimize()

        username = pynutil.insert("username: \"") + accepted_symbols + pynutil.insert("\"") + pynini.cross('@', ' ')
        domain_graph = accepted_symbols + pynini.accep('.') + accepted_symbols
        domain_graph = pynutil.insert("domain: \"") + domain_graph + pynutil.insert("\"")
        domain_common_graph = (
            pynutil.insert("domain: \"")
            + accepted_symbols
            + pynini.union(*accepted_common_domains)
            + pynutil.insert("\"")
        )

        protocol_start = pynini.accep("https://") | pynini.accep("http://")
        protocol_symbols = pynini.closure(
            (NEMO_ALPHA | pynutil.add_weight(graph_symbols | pynini.cross(":", "doppelpunkt"), -0.1))
            + pynutil.insert(" ")
        )
        protocol_end = pynini.accep("www.")
        protocol = protocol_start | protocol_end | (protocol_start + protocol_end)
        protocol = pynini.compose(protocol, protocol_symbols)
        protocol = pynutil.insert("protocol: \"") + protocol + pynutil.insert("\"")
        graph = username + domain_graph
        graph |= domain_common_graph
        graph |= protocol + pynutil.insert(" ") + domain_graph

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
