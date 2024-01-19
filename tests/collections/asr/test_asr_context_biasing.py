# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import os
from functools import lru_cache
import pytest
import torch
from nemo.collections.asr.parts.context_biasing.context_graph_ctc import ContextGraphCTC


class TestContextGraphCTC:
    @pytest.mark.unit
    def test_graph_building(self):
        context_biasing_list = [["gpu", [['▁g', 'p', 'u'], ['▁g', '▁p', '▁u']]]]
        context_graph = ContextGraphCTC(blank_id=1024)
        context_graph.build(context_biasing_list)
        assert context_graph.num_nodes == 8