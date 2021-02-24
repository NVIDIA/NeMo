# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import csv
import os

import pynini
from denormalization.data_loader_utils import get_abs_path
from denormalization.graph_utils import NEMO_NOT_SPACE, GraphFst, convert_space
from pynini.lib import pynutil


class WhiteListFst(GraphFst):
    def __init__(self):
        super().__init__(name="whitelist", kind="classify")

        whitelist = pynini.string_file(get_abs_path("data/whitelist.tsv")).invert()
        whitelist = pynutil.add_weight(whitelist, weight=-10)
        graph = pynutil.insert("name: \"") + whitelist + pynutil.insert("\"")
        self.fst = graph.optimize()
