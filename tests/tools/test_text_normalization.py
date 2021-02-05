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

import os
from unittest import TestCase

import numpy as np
import pytest
from tools.text_normalization.normalize import normalize_identity, normalize_nemo


class TestTextNormalization(TestCase):
    @pytest.mark.unit
    def test_identity(self):
        text_in = ["1", "12kg", "1 2"]
        text_pred = normalize_identity(text_in)
        text_norm = normalize_nemo(text_in)
        self.assertTrue(text_in == text_pred)
        self.assertTrue(text_norm != text_pred)

    @pytest.mark.unit
    def test_numbers(self):
        text_in = ["1", "12kg", "1, 2, 3"]
        text_gold = ["one", "twelve kilograms", "one, two, three"]
        text_out = normalize_nemo(text_in)
        self.assertTrue(text_out == text_gold)

    @pytest.mark.unit
    def test_time(self):
        text_valid = ["01:00", "01:00 am", "01:00 a.m.", "1.59 p.m."]
        text_gold = ["one o'clock", "one a m", "one a m", "one fifty nine p m"]
        text_invalid = ["1.60 p.m."]
        text_out_valid = normalize_nemo(text_valid)
        text_out_invalid = normalize_nemo(text_invalid)
        self.assertTrue(all(np.asarray(text_out_valid) == np.asarray(text_gold)))
        self.assertTrue(all(np.asarray(text_out_invalid) != np.asarray(text_gold)))

    @pytest.mark.unit
    def test_whitelist(self):
        text_in = ["Dr. Evil", "idea", "Mrs. Norris"]
        text_gold = ["Doctor Evil", "idea", "Misses Norris"]
        text_out = normalize_nemo(text_in)
        self.assertTrue(text_out == text_gold)

    @pytest.mark.unit
    def test_boundaries(self):
        text_valid = [" 1 ", "1", "1, ", "1!!!!", "(1)Hello"]
        text_gold_valid = list(map(lambda x: x.replace("1", "one"), text_valid))
        text_invalid = ["!1", "1!Hello"]
        text_out_valid = normalize_nemo(text_valid)
        text_out_invalid = normalize_nemo(text_invalid)
        self.assertTrue(all(np.asarray(text_invalid) == np.asarray(text_out_invalid)))
        self.assertTrue(all(np.asarray(text_gold_valid) == np.asarray(text_out_valid)))
