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
from parameterized import parameterized
from utils import PYNINI_AVAILABLE, parse_test_case_file


class TestRuCardinal:
    inverse_normalizer = InverseNormalizer() if PYNINI_AVAILABLE else None

    @parameterized.expand(parse_test_case_file('data_inverse_text_normalization/test_ru_cases_cardinal.txt'))
    @pytest.mark.skipif(
        not PYNINI_AVAILABLE, reason="`pynini` not installed, please install via nemo_text_processing/setup.sh"
    )
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    def test_denorm(self, test_input, expected):
        pred = self.inverse_normalizer.inverse_normalize(test_input, verbose=False)
        assert pred == expected


if __name__ == '__main__':
    from tqdm import tqdm
    import re

    def _del_space_separator(text):
        """
        10 000 -> 10000
        """
        pattern = re.compile(r'\d{1,3}(?=(?:\s\d{3})*$)')
        match = pattern.findall(text)
        if len(match) > 1:
            text_reformatted = ''.join(match)
            assert text_reformatted.replace(' ', '') == text.replace(' ', '')
            return text_reformatted
        return text

    def _del_spaces(text):
        # TODO: fix digits case
        return text.replace(" ", "")

    def _del_start_zero(text):
        """
        04 -> 4
        """
        if len(text) > 1 and text.startswith("0"):
            text = text[1:]
        return text

    inverse_normalizer = InverseNormalizer()
    # print(inverse_normalizer.inverse_normalize("миллион сто двадцать три тысячи", verbose=True)

    test_cases = parse_test_case_file('data_inverse_text_normalization/test_ru_cases_cardinal_last_file.txt')
    # test_cases = parse_test_case_file('./missing_ru_cases_cardinal.txt')
    count_correct = 0
    count_wrong = 0
    with open('missing_ru_cases_cardinal.txt', 'w') as f_out:
        with open('error_ru_cardinal.txt', 'w') as f:
            for test_input, expected in tqdm(test_cases):
                if test_input == 'sil':
                    continue

                pred = inverse_normalizer.inverse_normalize(test_input, verbose=False)
                if (
                    pred != _del_space_separator(expected)
                    and _del_spaces(pred) != _del_spaces(expected)
                    and _del_start_zero(expected) != pred
                ):
                    f.write(f'Input: {test_input}\n')
                    f.write(f'Truth: {expected}\n')
                    f.write(f'Pred : {pred}\n\n')
                    count_wrong += 1
                    f_out.write(f'{test_input}~{expected}\n')
                else:
                    count_correct += 1

    print(f'Correct: {count_correct} - {round(count_correct / (count_correct + count_wrong) * 100, 2)}%')
    print(f'Wrong: {count_wrong} - {round(count_wrong / (count_correct + count_wrong) * 100, 2)}%')
