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

from unittest import TestCase

from parameterized import parameterized
from tools.text_denormalization.denormalize import denormalize
from utils import parse_test_case_file


class TestDate(TestCase):
    @parameterized.expand(parse_test_case_file('data_text_denormalization/test_cases_date.txt'))
    def test_denorm(self, test_input, expected):
        pred = denormalize(test_input, verbose=False)
        assert pred == expected
