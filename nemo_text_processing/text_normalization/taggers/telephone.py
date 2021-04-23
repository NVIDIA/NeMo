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
from nemo_text_processing.text_normalization.graph_utils import GraphFst, delete_space, insert_space

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone
    """

    def __init__(self):
        super().__init__(name="telephone", kind="classify")
        # country code (excluding '+'), eg. 44 for UK.
        # +*** -  country code, optional
        # number part ***-***-****, or (***) ***-****
        # extension  1-9999 optional

        digit = pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv"))).optimize() | pynini.cross(
            "0", "o"
        )

        country_code = (
            pynutil.insert("country_code: \"")
            + pynutil.delete("+")
            + pynini.closure(digit + insert_space, 0, 2)
            + digit
            + pynutil.insert("\"")
        )
        optional_country_code = pynini.closure(
            country_code + pynini.closure(pynutil.delete("-"), 0, 1) + delete_space + insert_space, 0, 1
        )
        number_part = (
            (
                (pynini.closure(digit + insert_space, 2, 2) + digit + pynutil.delete("-"))
                | (
                    pynutil.delete("(")
                    + pynini.closure(digit + insert_space, 2, 2)
                    + digit
                    + pynutil.delete(")")
                    + pynini.closure(pynutil.delete("-"), 0, 1)
                    + delete_space
                )
            )
            + pynini.closure(insert_space + digit, 3, 3)
            + pynutil.delete("-")
            + pynini.closure(insert_space + digit, 4, 4)
        )
        number_part = pynutil.insert("number_part: \"") + number_part + pynutil.insert("\"")
        extension = (
            pynutil.insert("extension : \"")
            + pynini.closure(digit + insert_space, 0, 3)
            + digit
            + pynutil.insert("\"")
        )
        optional_extension = pynini.closure(insert_space + pynutil.delete("-") + extension, 0, 1)
        # import ipdb; ipdb.set_trace()

        graph = optional_country_code + number_part + optional_extension
        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
