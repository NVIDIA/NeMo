# ! /usr/bin/python
# -*- coding: utf-8 -*-

# =============================================================================
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
# =============================================================================

from unittest import TestCase

import pytest
from ruamel.yaml import YAML

from nemo.core import AppState


class TestAppState(TestCase):
    @pytest.mark.unit
    def test_shared_graph(self):
        # Create first instance of AppState.
        x = AppState()
        x.test_value = "ala"
        # Create second instance of AppState and test value.
        y = AppState()
        self.assertEqual(y.test_value, "ala")
        # Change second instance and test first one.
        y.test_value = "ola"
        self.assertEqual(x.test_value, "ola")
