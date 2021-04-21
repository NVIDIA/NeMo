# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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


from nemo_text_processing.text_normalization.data_loader_utils import get_abs_path
from nemo_text_processing.text_normalization.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
)

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
    delete_space = pynutil.delete(" ")
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class EmailFst(GraphFst):
    """
    Finite state transducer for classifying cardinals
        e.g. a.smith@gmail.com -> email { username: "a.smith" servername: "gmail" domain: "com" }
    """

    def __init__(self):
        super().__init__(name="email", kind="classify")

        graph = (
                pynutil.insert("username: \"") + pynini.accep('@gmail') + pynutil.insert("\"")
        )
        #
        #
        #
        # self.graph = pynini.invert(graph) @ delete_extra_spaces
        # self.graph = self.graph.optimize()
        #
        # optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", "\"true\" "), 0, 1)
        #
        # final_graph = optional_minus_graph + pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        #
        # final_graph = self.add_tokens(final_graph)
        # self.fst = final_graph.optimize()
