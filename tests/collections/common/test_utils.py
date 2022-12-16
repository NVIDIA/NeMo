# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import tempfile

import pytest

from nemo.collections.common.parts.utils import flatten, get_num_lines


class TestListUtils:
    @pytest.mark.unit
    def test_flatten(self):
        """Test flattening an iterable with different values: str, bool, int, float, complex.
        """
        test_cases = []
        test_cases.append({'input': ['aa', 'bb', 'cc'], 'golden': ['aa', 'bb', 'cc']})
        test_cases.append({'input': ['aa', ['bb', 'cc']], 'golden': ['aa', 'bb', 'cc']})
        test_cases.append({'input': ['aa', [['bb'], [['cc']]]], 'golden': ['aa', 'bb', 'cc']})
        test_cases.append({'input': ['aa', [[1, 2], [[3]], 4]], 'golden': ['aa', 1, 2, 3, 4]})
        test_cases.append({'input': [True, [2.5, 2.0 + 1j]], 'golden': [True, 2.5, 2.0 + 1j]})

        for n, test_case in enumerate(test_cases):
            assert flatten(test_case['input']) == test_case['golden'], f'Test case {n} failed!'


class TestTextUtils:
    @pytest.mark.unit
    def test_get_num_lines(self):
        """Test getting the number of lines from a text file.
        """
        test_cases = [0, 1, 10, 42, 97]

        for num_lines in test_cases:

            lines = [f'line {n}\n' for n in range(num_lines)]

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_file = os.path.join(tmp_dir, 'file.txt')
                with open(tmp_file, 'w') as f:
                    f.writelines(lines)

                assert get_num_lines(tmp_file) == num_lines, f'Failed for num_lines {num_lines}'
