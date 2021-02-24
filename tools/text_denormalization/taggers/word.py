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


class WordFst(GraphFst):
    def __init__(self):
        super().__init__(name="word", kind="classify")

        exceptions = pynutil.add_weight(
            pynini.string_file(get_abs_path("grammars/sentence_boundary_exceptions.txt")), weight=-10
        )
        word = (
            pynutil.insert("name: \"")
            + (pynini.closure(pynutil.add_weight(NEMO_NOT_SPACE, weight=1), 1) | convert_space(exceptions))
            + pynutil.insert("\"")
        )
        self.fst = word.optimize()
