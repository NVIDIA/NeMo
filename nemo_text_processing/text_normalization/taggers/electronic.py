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


from nemo_text_processing.text_normalization.graph_utils import NEMO_CHAR, NEMO_SIGMA, GraphFst

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
    delete_space = pynutil.delete(" ")
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: as URLs, email addresses, etc.
        e.g. a.smith@gmail.com -> electronic { username: "a.smith" servername: "gmail" domain: "com" }
    """

    def __init__(self):
        super().__init__(name="electronic", kind="classify")

        # protocol, username, password, domain,port, path, query_string, fragment_id     protocol://username:password@domain:port/path?query_string#fragment_id

        username = pynutil.insert("username: \"") + NEMO_SIGMA + pynutil.insert("\"") + pynini.cross('@', ' ')
        domain_graph = pynutil.insert("domain: \"") + NEMO_SIGMA + pynutil.insert("\"")
        graph = username + domain_graph

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
