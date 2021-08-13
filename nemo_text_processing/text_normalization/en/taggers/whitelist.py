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

from nemo_text_processing.text_normalization.en.graph_utils import GraphFst, convert_space
from nemo_text_processing.text_normalization.en.utils import get_abs_path, load_labels

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelist, e.g.
        misses -> tokens { name: "mrs" }
        for non-deterministic case: "Dr. Abc" ->
            tokens { name: "drive" } tokens { name: "Abc" }
            tokens { name: "doctor" } tokens { name: "Abc" }
            tokens { name: "Dr." } tokens { name: "Abc" }
    This class has highest priority among all classifier grammars. Whitelisted tokens are defined and loaded from "data/whitelist.tsv".

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, input_case: str, deterministic: bool = True):
        super().__init__(name="whitelist", kind="classify", deterministic=deterministic)

        def _get_whitelist_graph(input_case, file="data/whitelist.tsv"):
            whitelist = load_labels(get_abs_path(file))
            if input_case == "lower_cased":
                whitelist = [(x.lower(), y) for x, y in whitelist]
            else:
                whitelist = [(x, y) for x, y in whitelist]
            graph = pynini.string_map(whitelist)
            return graph

        def _get_whitelist_non_deterministic_graph(file="data/whitelist_alternatives.tsv"):
            whitelist = load_labels(get_abs_path(file))
            whitelist_lower = [(x.lower(), y.lower()) for x, y in whitelist]
            whitelist_cased = [(x, y) for x, y in whitelist]
            graph = pynini.string_map(whitelist_lower + whitelist_cased)
            return graph

        graph = _get_whitelist_graph(input_case)
        if not deterministic:
            graph |= _get_whitelist_graph("lower_cased") | _get_whitelist_non_deterministic_graph()

        self.graph = (convert_space(graph)).optimize()
        self.fst = (pynutil.insert("name: \"") + self.graph + pynutil.insert("\"")).optimize()
