# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_text_processing.text_normalization.de.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    delete_preserve_order,
)

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic, e.g.
        time { hours: "2" minutes: "15"} -> "zwei uhr fünfzehn"
        time { minutes: "15" hours: "2" } -> "viertel nach zwei"
        time { minutes: "15" hours: "2" } -> "fünfzehn nach zwei"
        time { hours: "14" minutes: "15"} -> "vierzehn uhr fünfzehn"
        time { minutes: "15" hours: "14" } -> "viertel nach zwei"
        time { minutes: "15" hours: "14" } -> "fünfzehn nach drei"
        time { minutes: "45" hours: "14" } -> "viertel vor drei"

    Args:
        cardinal: cardinal tagger GraphFst
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        night_to_early = pynini.invert(pynini.string_file(get_abs_path("data/time/hour_to_night.tsv"))).optimize()
        hour_to = pynini.invert(pynini.string_file(get_abs_path("data/time/hour_to.tsv"))).optimize()
        minute_to = pynini.invert(pynini.string_file(get_abs_path("data/time/minute_to.tsv"))).optimize()

        graph_zero = pynini.invert(pynini.string_file(get_abs_path("data/numbers/zero.tsv"))).optimize()
        number_verbalization = graph_zero | cardinal.two_digit_non_zero
        hour = pynutil.delete("hours: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        hour_verbalized = hour @ number_verbalization @ pynini.cdrewrite(
            pynini.cross("eins", "ein"), "[BOS]", "[EOS]", NEMO_SIGMA
        ) + pynutil.insert(" uhr")
        minute = pynutil.delete("minutes: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        zone = pynutil.delete("zone: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        optional_zone = pynini.closure(pynini.accep(" ") + zone, 0, 1)
        second = pynutil.delete("seconds: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        graph_hms = (
            hour_verbalized
            + pynini.accep(" ")
            + minute @ number_verbalization
            + pynutil.insert(" minuten")
            + pynini.accep(" ")
            + second @ number_verbalization
            + pynutil.insert(" sekunden")
            + optional_zone
            + delete_preserve_order
        )
        graph_hms @= pynini.cdrewrite(
            pynini.cross("eins minuten", "eine minute") | pynini.cross("eins sekunden", "eine sekunde"),
            pynini.union(" ", "[BOS]"),
            "",
            NEMO_SIGMA,
        )

        min_30 = [str(x) for x in range(1, 31)]
        min_30 = pynini.union(*min_30)
        min_29 = [str(x) for x in range(1, 30)]
        min_29 = pynini.union(*min_29)

        graph_h = hour_verbalized
        graph_hm = hour_verbalized + pynini.accep(" ") + minute @ number_verbalization

        graph_m_past_h = (
            minute @ min_30 @ (number_verbalization | pynini.cross("15", "viertel"))
            + pynini.accep(" ")
            + pynutil.insert("nach ")
            + hour @ pynini.cdrewrite(night_to_early, "[BOS]", "[EOS]", NEMO_SIGMA) @ number_verbalization
        )
        graph_m30_h = (
            minute @ pynini.cross("30", "halb")
            + pynini.accep(" ")
            + hour @ pynini.cdrewrite(night_to_early, "[BOS]", "[EOS]", NEMO_SIGMA) @ hour_to @ number_verbalization
        )
        graph_m_to_h = (
            minute @ minute_to @ min_29 @ (number_verbalization | pynini.cross("15", "viertel"))
            + pynini.accep(" ")
            + pynutil.insert("vor ")
            + hour @ pynini.cdrewrite(night_to_early, "[BOS]", "[EOS]", NEMO_SIGMA) @ hour_to @ number_verbalization
        )

        graph = (graph_hms | graph_h | graph_hm | graph_m_past_h | graph_m30_h | graph_m_to_h) + optional_zone
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
