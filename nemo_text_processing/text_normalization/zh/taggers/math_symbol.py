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
from nemo_text_processing.text_normalization.zh.graph_utils import GraphFst
from nemo_text_processing.text_normalization.zh.taggers.cardinal import Cardinal
from pynini.lib import pynutil


class MathSymbol(GraphFst):
    '''
        + -> tokens { sign: "加" }
    '''

    def __init__(self, deterministic: bool = True, lm: bool = False):
        super().__init__(name="sign", kind="classify", deterministic=deterministic)
        '''
            add your sign in data/math/symbol.tsv,this graph just convert sigh to character,you can add more 
            cases with detailed cases 
        '''
        score = (
            pynutil.insert("score: \"")
            + Cardinal().graph_cardinal
            + pynini.cross(":", "比")
            + Cardinal().graph_cardinal
            + pynutil.insert("\"")
        )
        graph = score
        self.fst = graph.optimize()
