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

from nemo_text_processing.text_normalization.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_space,
)

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        date { month: "february" day: "five" year: "twenty twelve" preserve_order: true } -> february fifth twenty twelve
        date { day: "five" month: "february" year: "twenty twelve" preserve_order: true } -> the fifth of february twenty twelve

    Args:
        ordinal: OrdinalFst
    """

    def __init__(self, ordinal: GraphFst):
        super().__init__(name="date", kind="verbalize")

        month = (
            pynutil.delete("month:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        day = (
            pynutil.delete("day:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        ) @ ordinal.suffix
        year = (
            pynutil.delete("year:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + delete_space
            + pynutil.delete("\"")
        )

        # month (day) year
        graph_mdy = (
            month + pynini.closure(delete_extra_space + day, 0, 1) + pynini.closure(delete_extra_space + year, 0, 1)
        )

        # day month year
        graph_dmy = (
            pynutil.insert("the ")
            + day
            + delete_extra_space
            + pynutil.insert("of ")
            + month
            + pynini.closure(delete_extra_space + year, 0, 1)
        )

        optional_preserve_order = pynini.closure(
            pynutil.delete("preserve_order:") + delete_space + pynutil.delete("true") + delete_space
            | pynutil.delete("field_order:")
            + delete_space
            + pynutil.delete("\"")
            + NEMO_NOT_QUOTE
            + pynutil.delete("\"")
            + delete_space
        )

        final_graph = (graph_mdy | year | graph_dmy) + delete_space + optional_preserve_order

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
