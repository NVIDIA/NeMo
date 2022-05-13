# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from nemo_text_processing.text_normalization.normalize import Normalizer
from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio
from parameterized import parameterized

import sys
sys.path.append("/home/ebakhturina/NeMo/tests/nemo_text_processing")
from utils import CACHE_DIR, PYNINI_AVAILABLE, parse_test_case_file


class TestPunctuation:
    normalizer_en = (
        Normalizer(input_case='cased', lang='en', cache_dir="/home/ebakhturina/NeMo/nemo_text_processing/text_normalization/cache_dir", overwrite_cache=False, post_process=True)
        if PYNINI_AVAILABLE
        else None
    )

    normalizer_en_no_post_process = (
        Normalizer(input_case='cased', lang='en',
                   cache_dir="/home/ebakhturina/NeMo/nemo_text_processing/text_normalization/cache_dir",
                   overwrite_cache=False, post_process=False)
        if PYNINI_AVAILABLE
        else None
    )

    normalizer_with_audio_en = (
        NormalizerWithAudio(input_case='cased', lang='en', cache_dir=CACHE_DIR, overwrite_cache=False)
        if PYNINI_AVAILABLE and CACHE_DIR and False
        else None
    )

    # address is tagged by the measure class
    @parameterized.expand(parse_test_case_file('en/data_text_normalization/test_cases_punctuation.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE, reason="`pynini` not installed, please install via nemo_text_processing/setup.sh"
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm(self, test_input, expected):
        pred = self.normalizer_en.normalize(test_input, verbose=True, punct_post_process=False)
        assert pred == expected, f"input: {test_input}"

        if self.normalizer_with_audio_en:
            pred_non_deterministic = self.normalizer_with_audio_en.normalize(
                test_input, n_tagged=30, punct_post_process=True
            )
            assert expected in pred_non_deterministic, f"input: {test_input}"


    @parameterized.expand(parse_test_case_file('en/data_text_normalization/test_cases_punctuation_match_input.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE, reason="`pynini` not installed, please install via nemo_text_processing/setup.sh"
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_norm_python_punct_post_process(self, test_input, expected):
        pred = self.normalizer_en_no_post_process.normalize(test_input, verbose=True, punct_post_process=True)
        assert pred == expected, f"input: {test_input}"


if __name__ == "__main__":
    print("----->", CACHE_DIR)

    tt = TestPunctuation()
    tt.test_norm()