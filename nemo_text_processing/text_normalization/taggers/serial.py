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

from nemo_text_processing.text_normalization.data_loader_utils import get_abs_path, load_labels
from nemo_text_processing.text_normalization.graph_utils import NEMO_ALPHA, GraphFst, insert_space

try:
    import pynini
    from pynini.lib import pynutil

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False


class SerialFst(GraphFst):
    """
    Finite state transducer for classifying serial.
        The serial is a combination of digits, letters and dashes, e.g.:
        c325 ->
        tokens { serial { value: "c three hundred twenty five" } }
        tokens { serial { value: "c three two five" } }
        tokens { serial { value: "c thirty two five" } }
        tokens { serial { value: "c three twenty five" } }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="serial", kind="classify", deterministic=deterministic)

        if deterministic:
            num_graph = cardinal.single_digits_graph
        else:
            num_graph = cardinal.graph

        alpha = NEMO_ALPHA
        if not deterministic:
            letter_pronunciation = pynini.string_map(load_labels(get_abs_path("data/letter_pronunciation.tsv")))
            alpha |= letter_pronunciation

        delimiter = insert_space | pynini.cross("-", " ")

        letter_num = pynini.closure(alpha + delimiter, 1) + num_graph
        num_letter = num_graph + pynini.closure(delimiter + alpha, 1)

        next_alpha_or_num = pynini.closure(delimiter + (alpha | num_graph))

        serial_graph = (letter_num | num_letter) + next_alpha_or_num
        graph = pynutil.insert("cardinal { integer: \"") + serial_graph

        if not deterministic:
            graph += pynini.closure(pynini.accep("s") | pynini.cross("s", "es"), 0, 1)

        graph += pynutil.insert("\" } units: \"serial\"")

        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
