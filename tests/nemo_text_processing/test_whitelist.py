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


import pytest
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
from nemo_text_processing.text_normalization.normalize import Normalizer
from parameterized import parameterized
from utils import PYNINI_AVAILABLE, parse_test_case_file


class TestWhitelist:
    inverse_normalizer = InverseNormalizer() if PYNINI_AVAILABLE else None

    @parameterized.expand(parse_test_case_file('data_inverse_text_normalization/test_cases_whitelist.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE, reason="`pynini` not installed, please install via nemo_text_processing/setup.sh"
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm(self, test_input, expected):
        pred = self.inverse_normalizer.inverse_normalize(test_input, verbose=False)
        assert pred == expected

    normalizer = Normalizer(input_case='lower_cased') if PYNINI_AVAILABLE else None

    @parameterized.expand(parse_test_case_file('data_text_normalization/test_cases_whitelist.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE, reason="`pynini` not installed, please install via nemo_text_processing/setup.sh"
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm(self, test_input, expected):
        pred = self.normalizer.normalize(test_input, verbose=False)
        assert pred == expected

    normalizer_uppercased = Normalizer(input_case='cased') if PYNINI_AVAILABLE else None
    cases_uppercased = {"Dr. Evil": "doctor Evil", "No. 4": "number four", "dr. Evil": "dr. Evil", "no. 4": "no. four"}

    @parameterized.expand(cases_uppercased.items())
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE, reason="`pynini` not installed, please install via nemo_text_processing/setup.sh"
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_cased(self, test_input, expected):
        pred = self.normalizer_uppercased.normalize(test_input, verbose=False)
        assert pred == expected
