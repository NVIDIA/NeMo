# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.es.graph_utils import articles_for_dates as articles
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    NEMO_SPACE,
    GraphFst,
    delete_preserve_order,
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
        date { day: "treinta y uno" month: "marzo" year: "dos mil" } -> "treinta y uno de marzo de dos mil"
        date { day: "uno" month: "mayo" year: "del mil nuevecientos noventa" } -> "primero de mayo del mil nuevecientos noventa"

    Args:
        ordinal: ordinal verbalizer GraphFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, ordinal: GraphFst, deterministic: bool = True):
        super().__init__(name="date", kind="verbalize", deterministic=deterministic)

        day_cardinal = pynutil.delete("day: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        if deterministic:
           day = day_cardinal @ pynini.cdrewrite(pynini.cross("un", "primero"), "[BOS]", "[EOS]", NEMO_SIGMA) # Traditional
        if not deterministic:
           day = day_cardinal @ pynini.cdrewrite(pynini.cross("un" , "uno"), "[BOS]", "[EOS]", NEMO_SIGMA) # Current in some parts
           day |= day_cardinal @ pynini.cdrewrite(pynini.cross("un", "primero"), "[BOS]", "[EOS]", NEMO_SIGMA) # Traditional


        month = pynutil.delete("month: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        year = pynutil.delete("year: \"") + articles + NEMO_SPACE + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        year = pynutil.add_weight(year, -0.001)

        # Insert preposition if wasn't originally with the year. This would mean a space was present
        year |= pynutil.delete("year: \"") + pynutil.insert("de ") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        # day month year
        graph_dmy = day + pynini.cross(NEMO_SPACE, " de ") + month + pynini.closure(pynini.accep(" ") + year, 0, 1)

        graph_mdy = month + NEMO_SPACE + day + pynini.closure(NEMO_SPACE + year, 0, 1)
        if deterministic:
           graph_mdy +=  pynutil.delete(" preserve_order: true") # Only accepts this if was explicitly passed     
        

        self.graph = graph_dmy |  graph_mdy
        final_graph = self.graph + delete_preserve_order

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
