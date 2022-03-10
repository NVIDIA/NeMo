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


import pytest
from nemo.collections.common.tokenizers.column_coder import FloatCode, IntCode, CategoryCode
import pandas as pd
import numpy as np
import string


class TestColumnCoder:

    @pytest.mark.unit
    def test_float(self):
        np.random.seed(1234)
        array = np.random.random(100)
        series = pd.Series(array)
        float_coder = FloatCode('t', 5, 0, 10)
        float_coder.compute_code(data_series=series, fillall=False)
        r = float_coder.encode('0.323')
        assert np.array_equal(np.array(r), np.array([37, 32, 27, 15, 1]))
        decoded = float_coder.decode(r)
        assert decoded == '0.32287'
        r = float_coder.encode('1.323')
        assert np.array_equal(np.array(r), np.array([37, 36, 24, 10, 1]))
        decoded = float_coder.decode(r)
        assert decoded == '0.90288'
        r = float_coder.encode('nan')
        assert np.array_equal(np.array(r), np.array([38, 36, 29, 19, 9]))
        decoded = float_coder.decode(r)
        assert decoded == 'nan'

        float_coder = FloatCode('t', 5, 0, 377)
        float_coder.compute_code(data_series=series, fillall=True)
        r = float_coder.encode('0.323')
        assert np.array_equal(np.array(r), np.array([1508, 1234, 1036, 613, 338]))
        decoded = float_coder.decode(r)
        assert decoded == '0.32299999995'
        r = float_coder.encode('nan')
        assert np.array_equal(np.array(r), np.array([1885, 1507, 1130, 753, 376]))
        decoded = float_coder.decode(r)
        assert decoded == 'nan'


    @pytest.mark.unit
    def test_int(self):
        np.random.seed(1234)
        array = np.random.randint(3, 1000, 100)
        int_coder = IntCode('i', 3, 0, 16)
        int_coder.compute_code(array, False)

        r = int_coder.encode('232')
        assert np.array_equal(np.array(r), np.array([32, 30, 2]))
        decoded = int_coder.decode(r)
        assert decoded == '232'

        r = int_coder.encode('nan')
        assert np.array_equal(np.array(r), np.array([36, 31, 15]))
        decoded = int_coder.decode(r)
        assert decoded == 'nan'

    @pytest.mark.unit
    def test_category(self):
        np.random.seed(1234)
        ALPHABET = np.array(list(string.ascii_lowercase + ' '))
        array = np.char.add(np.random.choice(ALPHABET, 1000), np.random.choice(ALPHABET, 1000))

        int_coder = CategoryCode('c', 0)
        int_coder.compute_code(array, False)

        r = int_coder.encode('xy')
        assert np.array_equal(np.array(r), np.array([509]))
        decoded = int_coder.decode(r)
        assert decoded == 'xy'

 