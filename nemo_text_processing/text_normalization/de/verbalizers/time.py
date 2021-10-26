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

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    delete_space,
    insert_space,
)

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing electronic
        e.g. time { hours: "два часа пятнадцать минут" } -> "два часа пятнадцать минут"

    Args:
        deterministic: if True will provide a single transduction option,
        for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        hour = pynutil.delete("hours: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        minute = pynutil.delete("minutes: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        zone = pynutil.delete("zone: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        optional_zone = pynini.closure(delete_space + insert_space + zone, 0, 1)
        second = pynutil.delete("seconds: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        graph_hms = (
            hour
            + pynutil.insert(" Uhr ")
            + delete_space
            + minute
            + pynutil.insert(" Minuten ")
            + delete_space
            + second
            + pynutil.insert(" Sekunden")
            + optional_zone
        )
        graph_hms @= pynini.cdrewrite(
            pynini.cross("eine Minuten", "eine Minute")
            | pynini.cross("eine Sekunden", "eine Sekunde")
            | pynini.cross("eine Stunden", "one Stunde"),
            pynini.union(" ", "[BOS]"),
            "",
            NEMO_SIGMA,
        )
        graph = hour + delete_space + insert_space + minute + optional_zone
        graph |= hour
        graph |= hour + delete_space + pynutil.insert(" uhr ")
        graph |= hour + delete_space + pynutil.insert(" uhr ") + minute + optional_zone
        graph |= graph_hms
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
