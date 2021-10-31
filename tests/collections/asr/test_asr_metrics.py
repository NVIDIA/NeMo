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

import io
import random
import string
from copy import deepcopy
from typing import List
from unittest.mock import Mock, patch

import pytest
import torch

from nemo.collections.asr.metrics.rnnt_wer import RNNTWER
from nemo.collections.asr.metrics.rnnt_wer_bpe import RNNTBPEWER
from nemo.collections.asr.metrics.wer import WER, word_error_rate
from nemo.collections.asr.metrics.wer_bpe import WERBPE
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.common.tokenizers import CharTokenizer


def build_char_tokenizer_with_vocabulary(vocabulary: List[str]) -> CharTokenizer:
    with patch('pathlib.Path.open', Mock(return_value=io.StringIO('\n'.join([repr(char) for char in vocabulary])))):
        char_tokenizer = CharTokenizer('a_path_which_will_not_be_used')
    # For some reason `WERBPE` takes vocabulary size of inner tokenizer. Mock inner tokenizer.
    setattr(char_tokenizer, "tokenizer", Mock(vocab_size=char_tokenizer.vocab_size))
    return char_tokenizer


class TestWordErrorRate:

    vocabulary = [' '] + list(string.ascii_lowercase) + ["'"]
    char_tokenizer = build_char_tokenizer_with_vocabulary(vocabulary)

    def __string_to_ctc_tensor(self, txt: str, use_tokenizer: bool) -> torch.Tensor:
        # This function emulates how CTC output could like for txt
        if use_tokenizer:
            blank_id = self.char_tokenizer.vocab_size
            string_in_id_form = self.char_tokenizer.text_to_ids(txt)
        else:
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

    def __reference_string_to_tensor(self, txt: str, use_tokenizer: bool) -> torch.Tensor:
        # Reference tensors aren't produced by CTC logic
        if use_tokenizer:
            string_in_id_form = self.char_tokenizer.text_to_ids(txt)
        else:
            char_to_ind = dict([(self.vocabulary[i], i) for i in range(len(self.vocabulary))])
            string_in_id_form = [char_to_ind[c] for c in txt]
        return torch.Tensor(string_in_id_form).unsqueeze(0)

    def get_wer(self, wer, prediction: str, reference: str, use_tokenizer: bool):
        predictions_tensor = self.__string_to_ctc_tensor(prediction, use_tokenizer)
        targets_tensor = self.__reference_string_to_tensor(reference, use_tokenizer)
        if wer.batch_dim_index > 0:
            targets_tensor.transpose_(0, 1)
            predictions_tensor.transpose_(0, 1)
        wer(predictions=predictions_tensor, targets=targets_tensor, target_lengths=torch.tensor([len(reference)]))
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
    @pytest.mark.parametrize("batch_dim_index", [0, 1])
    @pytest.mark.parametrize("test_wer_bpe", [False, True])
    def test_wer_metric_simple(self, batch_dim_index, test_wer_bpe):
        if test_wer_bpe:
            wer = WERBPE(self.char_tokenizer, batch_dim_index, use_cer=False, ctc_decode=True)
        else:
            wer = WER(vocabulary=self.vocabulary, batch_dim_index=batch_dim_index, use_cer=False, ctc_decode=True)

        assert self.get_wer(wer, 'cat', 'cot', test_wer_bpe) == 1.0
        assert self.get_wer(wer, 'gpu', 'g p u', test_wer_bpe) == 1.0
        assert self.get_wer(wer, 'g p u', 'gpu', test_wer_bpe) == 3.0
        assert self.get_wer(wer, 'ducati motorcycle', 'motorcycle', test_wer_bpe) == 1.0
        assert self.get_wer(wer, 'ducati motorcycle', 'ducuti motorcycle', test_wer_bpe) == 0.5
        assert abs(self.get_wer(wer, 'a f c', 'a b c', test_wer_bpe) - 1.0 / 3.0) < 1e-6

    @pytest.mark.unit
    @pytest.mark.parametrize("test_wer_bpe", [False, True])
    def test_wer_metric_randomized(self, test_wer_bpe):
        """This test relies on correctness of word_error_rate function."""

        def __random_string(length):
            return ''.join(random.choice(''.join(self.vocabulary)) for _ in range(length))

        if test_wer_bpe:
            wer = WERBPE(deepcopy(self.char_tokenizer), batch_dim_index=0, use_cer=False, ctc_decode=True)
        else:
            wer = WER(vocabulary=self.vocabulary, batch_dim_index=0, use_cer=False, ctc_decode=True)

        for test_id in range(256):
            n1 = random.randint(1, 512)
            n2 = random.randint(1, 512)
            s1 = __random_string(n1)
            s2 = __random_string(n2)
            # skip empty strings as reference
            if s2.strip():
                assert (
                    abs(
                        self.get_wer(wer, prediction=s1, reference=s2, use_tokenizer=test_wer_bpe)
                        - word_error_rate(hypotheses=[s1], references=[s2])
                    )
                    < 1e-6
                )

    @pytest.mark.unit
    @pytest.mark.parametrize("test_wer_bpe", [False, True])
    def test_wer_metric_decode(self, test_wer_bpe):
        if test_wer_bpe:
            wer = WERBPE(self.char_tokenizer, batch_dim_index=0, use_cer=False, ctc_decode=True)
        else:
            wer = WER(vocabulary=self.vocabulary.copy(), batch_dim_index=0, use_cer=False, ctc_decode=True)

        tokens = self.__string_to_ctc_tensor('cat', use_tokenizer=test_wer_bpe)[0].int().numpy().tolist()
        assert tokens == [3, 1, 20]

        tokens_decoded = wer.decode_ids_to_tokens(tokens)
        assert tokens_decoded == ['c', 'a', 't']

        str_decoded = wer.decode_tokens_to_str(tokens)
        assert str_decoded == 'cat'

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_dim_index", [0, 1])
    @pytest.mark.parametrize("test_wer_bpe", [False, True])
    def test_wer_metric_return_hypothesis(self, batch_dim_index, test_wer_bpe):
        wer = WER(vocabulary=self.vocabulary.copy(), batch_dim_index=batch_dim_index, use_cer=False, ctc_decode=True)

        tensor = self.__string_to_ctc_tensor('cat', test_wer_bpe).int()
        if batch_dim_index > 0:
            tensor.transpose_(0, 1)

        # pass batchsize 1 tensor, get back list of length 1 Hypothesis
        hyp = wer.ctc_decoder_predictions_tensor(tensor, return_hypotheses=True)
        hyp = hyp[0]
        assert isinstance(hyp, Hypothesis)

        assert hyp.y_sequence is None
        assert hyp.score == -1.0
        assert hyp.text == 'cat'
        assert hyp.alignments == [3, 1, 20]
        assert hyp.length == 0

        length = torch.tensor([tensor.shape[1 - batch_dim_index]], dtype=torch.long)

        # pass batchsize 1 tensor, get back list of length 1 Hypothesis [add length info]
        hyp = wer.ctc_decoder_predictions_tensor(tensor, predictions_len=length, return_hypotheses=True)
        hyp = hyp[0]
        assert isinstance(hyp, Hypothesis)
        assert hyp.length == 3

    def decode_token_to_str_with_vocabulary_mock(self, ids):
        return ''.join([self.vocabulary[id_] for id_ in ids])

    def get_wer_rnnt(self, prediction: str, reference: str, batch_dim_index: int, test_wer_bpe: bool):
        rnnt_decoder_predictions_tensor_mock = Mock(return_value=([prediction], None))
        if test_wer_bpe:
            decoding = Mock(
                blank_id=self.char_tokenizer.tokenizer.vocab_size,
                tokenizer=deepcopy(self.char_tokenizer),
                rnnt_decoder_predictions_tensor=rnnt_decoder_predictions_tensor_mock,
                decode_tokens_to_str=self.char_tokenizer.ids_to_text,
            )
            wer = RNNTBPEWER(decoding, batch_dim_index=batch_dim_index, use_cer=False)
        else:
            decoding = Mock(
                blank_id=len(self.vocabulary),
                labels_map=self.vocabulary.copy(),
                rnnt_decoder_predictions_tensor=rnnt_decoder_predictions_tensor_mock,
                decode_tokens_to_str=self.decode_token_to_str_with_vocabulary_mock,
            )
            wer = RNNTWER(decoding, batch_dim_index=batch_dim_index, use_cer=False)
        targets_tensor = self.__reference_string_to_tensor(reference, test_wer_bpe)
        if wer.batch_dim_index > 0:
            targets_tensor.transpose_(0, 1)
        wer(
            encoder_output=None,
            encoded_lengths=None,
            targets=targets_tensor,
            target_lengths=torch.tensor([len(reference)]),
        )
        res, _, _ = wer.compute()
        res = res.detach().cpu()
        # return res[0] / res[1]
        return res.item()

    @pytest.mark.unit
    @pytest.mark.parametrize("batch_dim_index", [0, 1])
    @pytest.mark.parametrize("test_wer_bpe", [False, True])
    def test_rnnt_wer_metric_simple(self, batch_dim_index, test_wer_bpe):
        assert self.get_wer_rnnt('cat', 'cot', batch_dim_index, test_wer_bpe) == 1.0
        assert self.get_wer_rnnt('gpu', 'g p u', batch_dim_index, test_wer_bpe) == 1.0
        assert self.get_wer_rnnt('g p u', 'gpu', batch_dim_index, test_wer_bpe) == 3.0
        assert self.get_wer_rnnt('ducati motorcycle', 'motorcycle', batch_dim_index, test_wer_bpe) == 1.0
        assert self.get_wer_rnnt('ducati motorcycle', 'ducuti motorcycle', batch_dim_index, test_wer_bpe) == 0.5
        assert abs(self.get_wer_rnnt('a f c', 'a b c', batch_dim_index, test_wer_bpe) - 1.0 / 3.0) < 1e-6

    @pytest.mark.unit
    @pytest.mark.parametrize("test_wer_bpe", [False, True])
    def test_rnnt_wer_metric_randomized(self, test_wer_bpe):
        """This test relies on correctness of word_error_rate function."""

        def __random_string(length):
            return ''.join(random.choice(''.join(self.vocabulary)) for _ in range(length))

        for test_id in range(256):
            n1 = random.randint(1, 512)
            n2 = random.randint(1, 512)
            s1 = __random_string(n1)
            s2 = __random_string(n2)
            # skip empty strings as reference
            if s2.strip():
                assert (
                    abs(
                        self.get_wer_rnnt(prediction=s1, reference=s2, batch_dim_index=0, test_wer_bpe=test_wer_bpe)
                        - word_error_rate(hypotheses=[s1], references=[s2])
                    )
                    < 1e-6
                )
