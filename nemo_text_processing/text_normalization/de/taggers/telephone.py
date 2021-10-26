# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
    NEMO_DIGIT,
    NEMO_SIGMA,
    GraphFst,
    delete_extra_space,
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


class TelephoneFst(GraphFst):
    """
    Finite state transducer for classifying telephone, which includes country code, number part and extension 

    E.g 
    "8-913-983-56-01" -> telephone { number_part: "восемь девятьсот тринадцать девятьсот восемьдесят три пятьдесят шесть ноль один" }

    Args:
        number_names: number_names for cardinal and ordinal numbers
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="telephone", kind="classify", deterministic=deterministic)

        digit = (
            pynini.invert(pynini.string_file(get_abs_path("data/numbers/digit.tsv"))).optimize()
            | pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv"))).optimize()
        )

        country_code = (
            pynutil.insert("country_code: \"")
            + pynini.closure(pynini.cross("+", "plus "), 0, 1)
            + (NEMO_DIGIT + NEMO_DIGIT) @ cardinal.graph_hundred_component_at_least_one_none_zero_digit
            + pynutil.insert("\"")
        )
        optional_country_code = pynini.closure(
            country_code + pynini.closure(pynutil.delete("-"), 0, 1) + delete_space + insert_space, 0, 1
        )

        del_separator = pynini.closure(pynini.union("-", " "), 0, 1)

        numbers = pynini.closure(
            (NEMO_DIGIT | (NEMO_DIGIT + NEMO_DIGIT)) @ cardinal.graph_hundred_component_at_least_one_none_zero_digit
            + insert_space,
            1,
        )

        numbers = (numbers + del_separator) | (
            pynutil.delete("(") + numbers + (pynutil.delete(") ") | pynutil.delete(")-")) + numbers
        )

        number_length = pynini.closure((NEMO_DIGIT | pynini.union("-", " ", ")", "(")), 7)
        number_part = pynini.compose(number_length, numbers) @ pynini.cdrewrite(delete_extra_space, "", "", NEMO_SIGMA)
        number = pynutil.insert("number_part: \"") + number_part + pynutil.insert("\"")

        graph = optional_country_code + number

        final_graph = self.add_tokens(graph)
        self.fst = final_graph.optimize()
