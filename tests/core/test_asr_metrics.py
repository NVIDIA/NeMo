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


import random
import string

import pytest
import torch

from nemo.collections.asr.metrics.wer import WER, word_error_rate


class TestWordErrorRate:

    vocabulary = [
        " ",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "'",
    ]

    def __string_to_tensor(self, txt: str) -> torch.Tensor:
        char_to_ind = dict([(self.vocabulary[i], i) for i in range(len(self.vocabulary))])
        return torch.Tensor([char_to_ind[c] for c in txt]).unsqueeze(0)

    @pytest.mark.unit
    def test_wer_function(self):
        assert word_error_rate(hypotheses=['cat'], references=['cot']) == 1.0
        assert word_error_rate(hypotheses=['GPU'], references=['G P U']) == 1.0
        assert word_error_rate(hypotheses=['G P U'], references=['GPU']) == 3.0
        assert word_error_rate(hypotheses=['ducati motorcycle'], references=['motorcycle']) == 1.0
        assert word_error_rate(hypotheses=['ducati motorcycle'], references=['ducuti motorcycle']) == 0.5
        assert word_error_rate(hypotheses=['a B c'], references=['a b c']) == 1.0 / 3.0

    @pytest.mark.unit
    def test_wer_metric_simple(self):
        wer = WER(vocabulary=self.vocabulary, batch_dim_index=0, use_cer=False, ctc_decode=True)

        def get_wer(prediction: str, reference: str):
            res = (
                wer(
                    predictions=self.__string_to_tensor(prediction),
                    targets=self.__string_to_tensor(reference),
                    target_lengths=torch.tensor([len(reference)]),
                )
                .detach()
                .cpu()
            )
            return res[0] / res[1]

        assert get_wer('cat', 'cot') == 1.0
        assert get_wer('gpu', 'g p u') == 1.0
        assert get_wer('g p u', 'gpu') == 3.0
        assert get_wer('ducati motorcycle', 'motorcycle') == 1.0
        assert get_wer('ducati motorcycle', 'ducuti motorcycle') == 0.5
        assert get_wer('a f c', 'a b c') == 1.0 / 3.0

    @pytest.mark.unit
    @pytest.mark.pleasefixme
    def test_wer_metric_randomized(self):
        """This test relies on correctness of word_error_rate function"""

        def __randomString(N):
            return ''.join(random.choice(string.ascii_lowercase + ' ') for i in range(N))

        wer = WER(vocabulary=self.vocabulary, batch_dim_index=0, use_cer=False, ctc_decode=True)

        def get_wer(prediction: str, reference: str):
            res = (
                wer(
                    predictions=self.__string_to_tensor(prediction),
                    targets=self.__string_to_tensor(reference),
                    target_lengths=torch.tensor([len(reference)]),
                )
                .detach()
                .cpu()
            )
            return res[0] / res[1]

        for test_id in range(256):
            n1 = random.randint(0, 512)
            n2 = random.randint(0, 512)
            s1 = __randomString(n1)
            s2 = __randomString(n2)

            assert get_wer(prediction=s1, reference=s2) == word_error_rate(hypotheses=[s1], references=[s2])

            # @Oleksii: THIS TEST IS UNSTABLE!

            # One failed case:
            # assert get_wer(prediction=s1, reference=s2) == word_error_rate(hypotheses=[s1], references=[s2])
            # assert tensor(4.3333) == 4.666666666666667
            # -tensor(4.3333)
            # +4.666666666666667

            # Second failed case:
            # assert get_wer(prediction=s1, reference=s2) == word_error_rate(hypotheses=[s1], references=[s2])
            # assert tensor(2.1818) == 2.272727272727273
            # -tensor(2.1818)
            # +2.272727272727273
