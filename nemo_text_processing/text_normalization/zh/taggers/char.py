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
import pynini
from nemo_text_processing.text_normalization.zh.graph_utils import (
    NEMO_ALPHA,
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_NOT_SPACE,
    NEMO_PUNCT,
    GraphFst,
)
from nemo_text_processing.text_normalization.zh.utils import get_abs_path, load_labels
from pynini.lib import pynutil, utf8


class Char(GraphFst):
    '''
        你 -> tokens { char: "你" }
    '''

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="char", kind="classify", deterministic=deterministic)

        graph = pynutil.insert("name: \"") + NEMO_CHAR + pynutil.insert("\"")
        self.fst = graph.optimize()
