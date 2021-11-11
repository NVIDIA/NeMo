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

from nemo_text_processing.text_normalization.de.utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_CHAR,
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
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
        tokens { date { day: "первое мая" } } -> "первое мая"

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, ordinal: GraphFst, deterministic: bool = True):
        super().__init__(name="date", kind="verbalize", deterministic=deterministic)

        day_cardinal = pynutil.delete("day: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        day = day_cardinal @ pynini.cdrewrite(ordinal.ordinal_stem, "", "[EOS]", NEMO_SIGMA) + pynutil.insert("ter")

        months_names = pynini.union(*[x[1] for x in load_labels(get_abs_path("data/months/abbr_to_name.tsv"))])
        month = pynutil.delete("month: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        final_month = month @ months_names
        final_month |= month @ pynini.difference(NEMO_SIGMA, months_names) @ pynini.cdrewrite(
            ordinal.ordinal_stem, "", "[EOS]", NEMO_SIGMA
        ) + pynutil.insert("ter")

        year = pynutil.delete("year: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        # day month year
        graph_dmy = day + pynini.accep(" ") + final_month + pynini.closure(pynini.accep(" ") + year, 0, 1)

        optional_preserve_order = pynini.closure(
            pynutil.delete(" preserve_order: true")
            | (pynutil.delete(" field_order: \"") + NEMO_NOT_QUOTE + pynutil.delete("\""))
        )

        final_graph = graph_dmy + optional_preserve_order
        final_graph |= year

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
