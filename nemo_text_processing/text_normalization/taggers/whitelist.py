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

from nemo_text_processing.text_normalization.data_loader_utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.graph_utils import GraphFst, convert_space

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
    This class has highest priority among all classifier grammars. Whitelisted tokens are defined and loaded from "data/whitelist.tsv".

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, input_case: str):
        super().__init__(name="whitelist", kind="classify")

        whitelist = load_labels(get_abs_path("data/whitelist.tsv"))
        if input_case == "lower_cased":
            whitelist = [(x.lower(), y) for x, y in whitelist]
        else:
            whitelist = [(x, y) for x, y in whitelist]

        graph = pynini.string_map(whitelist)

        graph = pynutil.insert("name: \"") + convert_space(graph) + pynutil.insert("\"")
        self.fst = graph.optimize()
