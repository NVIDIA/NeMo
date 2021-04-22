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

from nemo_text_processing.text_normalization.data_loader_utils import load_labels
from nemo_text_processing.text_normalization.graph_utils import NEMO_NOT_SPACE, GraphFst, convert_space

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class WordFst(GraphFst):
    """
    Finite state transducer for classifying word. Considers sentence boundary exceptions.
        e.g. sleep -> tokens { name: "sleep" }

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
    """

    def __init__(self, input_case: str):
        super().__init__(name="word", kind="classify")
        exceptions = load_labels("data/sentence_boundary_exceptions.txt")
        if input_case == "lower_cased":
            exceptions = [x.lower() for x, in exceptions]
        else:
            exceptions = [x for x, in exceptions]

            
        exceptions = pynini.string_map(exceptions)

        word = (
            pynutil.insert("name: \"")
            + (pynini.closure(pynutil.add_weight(NEMO_NOT_SPACE, weight=0.1), 1) | convert_space(exceptions))
            + pynutil.insert("\"")
        )
        self.fst = word.optimize()
