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
from nemo.collections.asr.parts.rnnt_utils import Hypothesis
from nemo.utils import logging


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

    def __string_to_ctc_tensor(self, txt: str) -> torch.Tensor:
        # This function emulates how CTC output could like for txt
        blank_id = len(self.vocabulary)
        char_to_ind = dict([(self.vocabulary[i], i) for i in range(len(self.vocabulary))])
        string_in_id_form = [char_to_ind[c] for c in txt]
        ctc_list = []
        prev_id = -1
        for c in string_in_id_form:
            # when character is repeated we need to insert CTC blank symbol
            if c != prev_id:
                ctc_list.append(c)
            else:
                ctc_list.append(blank_id)
                ctc_list.append(c)
            prev_id = c
        return torch.Tensor(ctc_list).unsqueeze(0)

    def __reference_string_to_tensor(self, txt: str) -> torch.Tensor:
        # Reference tensors aren't produced by CTC logic
        char_to_ind = dict([(self.vocabulary[i], i) for i in range(len(self.vocabulary))])
        string_in_id_form = [char_to_ind[c] for c in txt]
        return torch.Tensor(string_in_id_form).unsqueeze(0)

    def get_wer(self, wer, prediction: str, reference: str):
        wer(
            predictions=self.__string_to_ctc_tensor(prediction),
            targets=self.__reference_string_to_tensor(reference),
            target_lengths=torch.tensor([len(reference)]),
        )
        res, _, _ = wer.compute()
        res = res.detach().cpu()
        # return res[0] / res[1]
        return res.item()

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

        assert self.get_wer(wer, 'cat', 'cot') == 1.0
        assert self.get_wer(wer, 'gpu', 'g p u') == 1.0
        assert self.get_wer(wer, 'g p u', 'gpu') == 3.0
        assert self.get_wer(wer, 'ducati motorcycle', 'motorcycle') == 1.0
        assert self.get_wer(wer, 'ducati motorcycle', 'ducuti motorcycle') == 0.5
        assert abs(self.get_wer(wer, 'a f c', 'a b c') - 1.0 / 3.0) < 1e-6

    @pytest.mark.unit
    def test_wer_metric_randomized(self):
        """This test relies on correctness of word_error_rate function."""

        def __randomString(N):
            return ''.join(random.choice(''.join(self.vocabulary)) for i in range(N))

        wer = WER(vocabulary=self.vocabulary, batch_dim_index=0, use_cer=False, ctc_decode=True)

        for test_id in range(256):
            n1 = random.randint(1, 512)
            n2 = random.randint(1, 512)
            s1 = __randomString(n1)
            s2 = __randomString(n2)
            # skip empty strings as reference
            if s2.strip():
                assert (
                    abs(
                        self.get_wer(wer, prediction=s1, reference=s2)
                        - word_error_rate(hypotheses=[s1], references=[s2])
                    )
                    < 1e-6
                )

    @pytest.mark.unit
    def test_wer_metric_decode(self):
        wer = WER(vocabulary=self.vocabulary, batch_dim_index=0, use_cer=False, ctc_decode=True)

        tokens = self.__string_to_ctc_tensor('cat')[0].int().numpy().tolist()
        assert tokens == [3, 1, 20]

        tokens_decoded = wer.decode_ids_to_tokens(tokens)
        assert tokens_decoded == ['c', 'a', 't']

        str_decoded = wer.decode_tokens_to_str(tokens)
        assert str_decoded == 'cat'

    @pytest.mark.unit
    def test_wer_metric_return_hypothesis(self):
        wer = WER(vocabulary=self.vocabulary, batch_dim_index=0, use_cer=False, ctc_decode=True)

        tensor = self.__string_to_ctc_tensor('cat').int()

        # pass batchsize 1 tensor, get back list of length 1 Hypothesis
        hyp = wer.ctc_decoder_predictions_tensor(tensor, return_hypotheses=True)
        hyp = hyp[0]
        assert isinstance(hyp, Hypothesis)

        assert hyp.y_sequence is None
        assert hyp.score == -1.0
        assert hyp.text == 'cat'
        assert hyp.alignments == [3, 1, 20]
        assert hyp.length == 0

        length = torch.tensor([tensor.shape[-1]], dtype=torch.long)

        # pass batchsize 1 tensor, get back list of length 1 Hypothesis [add length info]
        hyp = wer.ctc_decoder_predictions_tensor(tensor, predictions_len=length, return_hypotheses=True)
        hyp = hyp[0]
        assert isinstance(hyp, Hypothesis)
        assert hyp.length == 3
