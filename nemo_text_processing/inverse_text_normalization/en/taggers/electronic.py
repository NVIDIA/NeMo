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

from nemo_text_processing.inverse_text_normalization.en.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_ALPHA, GraphFst, insert_space

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class ElectronicFst(GraphFst):
    """
    Finite state transducer for classifying electronic: as URLs, email addresses, etc.
        e.g. c d f one at a b c dot e d u -> tokens { electronic { username: "cdf1" domain: "abc.edu" } }
    """

    def __init__(self):
        super().__init__(name="electronic", kind="classify")

        delete_extra_space = pynutil.delete(" ")
        alpha_num = (
            NEMO_ALPHA
            | pynini.string_file(get_abs_path("data/numbers/digit.tsv"))
            | pynini.string_file(get_abs_path("data/numbers/zero.tsv"))
        )
        username = (
            pynutil.insert("username: \"")
            + pynini.closure(
                alpha_num + delete_extra_space + pynini.closure(pynini.cross("dot", '.') + delete_extra_space, 0, 1)
            )
            + alpha_num
            + pynutil.insert("\"")
        )
        single_alphanum = pynini.closure(alpha_num + delete_extra_space) + alpha_num
        server = single_alphanum | pynini.string_file(get_abs_path("data/electronic/server_name.tsv"))
        domain = single_alphanum | pynini.string_file(get_abs_path("data/electronic/domain.tsv"))
        domain_graph = (
            pynutil.insert("domain: \"")
            + server
            + delete_extra_space
            + pynini.cross("dot", ".")
            + delete_extra_space
            + domain
            + pynutil.insert("\"")
        )
        graph = username + delete_extra_space + pynutil.delete("at") + insert_space + delete_extra_space + domain_graph

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
