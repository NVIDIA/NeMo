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
    NEMO_CHAR,
    NEMO_NOT_QUOTE,
    NEMO_NOT_SPACE,
    NEMO_SIGMA,
    GraphFst,
)
from nemo_text_processing.text_normalization.zh.utils import get_abs_path
from pynini.lib import pynutil


class Char(GraphFst):
    '''
        tokens { char: "你" } -> 你
    '''

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="char", kind="verbalize", deterministic=deterministic)

        graph = pynutil.delete("name: \"") + NEMO_NOT_QUOTE + pynutil.delete("\"")
        self.fst = graph.optimize()
