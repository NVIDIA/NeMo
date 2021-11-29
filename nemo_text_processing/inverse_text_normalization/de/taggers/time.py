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


from nemo_text_processing.inverse_text_normalization.de.utils import get_abs_path
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_SIGMA, GraphFst

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time
        e.g. acht uhr -> time { hours: "8" minutes: "00" }
        e.g. dreizehn uhr -> time { hours: "13" minutes: "00" }
        e.g. dreizehn uhr zehn -> time { hours: "13" minutes: "10" }
        e.g. acht uhr abends -> time { hours: "8" minutes: "00" suffix: "abends"}
        e.g. acht uhr nachmittags -> time { hours: "8" minutes: "00" suffix: "nachmittags"}
        e.g. viertel vor zwölf -> time { minutes: "45" hours: "11" }
        e.g. viertel nach zwölf -> time { minutes: "15" hours: "12" }
        e.g. halb zwölf -> time { minutes: "30" hours: "11" }
        e.g. viertel zwölf -> time { minutes: "15" hours: "11" }
        e.g. drei minuten vor zwölf -> time { minutes: "57" hours: "11" }
        e.g. drei vor zwölf -> time { minutes: "57" hours: "11" }
        e.g. drei minuten nach zwölf -> time { minutes: "03" hours: "12" }
        e.g. drei viertel zwölf -> time { minutes: "45" hours: "11" }
    """

    def __init__(self, tn_time: GraphFst):
        super().__init__(name="time", kind="classify")
        # hours, minutes, seconds, suffix, zone, style, speak_period
        # lazy way to make sure compounds work
        optional_delete_space = pynini.closure(NEMO_SIGMA | pynutil.delete(" ", weight=0.0001))
        graph = (tn_time.graph @ optional_delete_space).invert().optimize()
        self.fst = self.add_tokens(graph).optimize()
