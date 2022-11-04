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

from itertools import permutations

import numpy as np
import pytest
import torch

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.utils.diarization_utils import concat_perm_word_error_rate


def perm_concat_perm_word_error_rate(spk_hypothesis, spk_reference):
    """
    Calculate cpWER with actual permutations to see LSA algorithm shows the correct result.
    """
    p_wer_list, permed_hyp_lists = [], []
    ref_word_list = []

    # Concatenate the hypothesis transcripts into a list
    for spk_id, word_list in enumerate(spk_reference):
        ref_word_list.append(word_list)
    ref_trans = " ".join(ref_word_list)

    # Calculate WER for every permutation
    for hyp_word_list in permutations(spk_hypothesis):
        hyp_trans = " ".join(hyp_word_list)
        permed_hyp_lists.append(hyp_trans)

        # Calculate a WER value of the permuted and concatenated transcripts
        p_wer = word_error_rate(hypotheses=[hyp_trans], references=[ref_trans])
        p_wer_list.append(p_wer)

    # Find the lowest WER and its hypothesis transcript
    argmin_idx = np.argmin(p_wer_list)
    min_perm_hyp_trans = permed_hyp_lists[argmin_idx]
    cpWER = p_wer_list[argmin_idx]
    return cpWER, min_perm_hyp_trans, ref_trans


def calculate_wer_count(_ins, _del, _sub, ref_word_count):
    return (_ins + _del + _sub) / ref_word_count


def word_count_ref(ref):
    return sum([len(w.split()) for w in ref])


def permuted_input_test(hyp, ref, calculated):
    """
    Randomly permute the input to see if evaluation result stays the same.
    """
    for hyp_permed in permutations(hyp):
        cpWER, hyp_min, ref_str = concat_perm_word_error_rate(spk_hypothesis=hyp, spk_reference=ref)
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6


class TestConcatMinPermWordErrorRate:
    @pytest.mark.unit
    def test_cpwer_oneword(self):
        hyp = ["oneword"]
        ref = ["oneword"]
        _ins, _del, _sub = 0, 0, 0
        cpWER, hyp_min, ref_str = concat_perm_word_error_rate(spk_hypothesis=hyp, spk_reference=ref)
        ref_word_count = word_count_ref(ref)
        calculated = calculate_wer_count(_ins, _del, _sub, ref_word_count)
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6
        permuted_input_test(hyp, ref, calculated)
        cpWER_perm, hyp_min_perm, ref_str = perm_concat_perm_word_error_rate(spk_hypothesis=hyp, spk_reference=ref)
        diff = torch.abs(torch.tensor(cpWER_perm - cpWER))
        assert diff <= 1e-6

        # Test with a substitution
        hyp = ["wrongword"]
        _ins, _del, _sub = 0, 0, 1
        cpWER, hyp_min, ref_str = concat_perm_word_error_rate(spk_hypothesis=hyp, spk_reference=ref)
        calculated = calculate_wer_count(_ins, _del, _sub, ref_word_count)
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6
        permuted_input_test(hyp, ref, calculated)
        cpWER_perm, hyp_min_perm, ref_str = perm_concat_perm_word_error_rate(spk_hypothesis=hyp, spk_reference=ref)
        diff = torch.abs(torch.tensor(cpWER_perm - cpWER))
        assert diff <= 1e-6

    @pytest.mark.unit
    def test_cpwer_perfect(self):
        hyp = ["aa bb cc", "dd ee", "ff"]
        ref = ["aa bb cc", "dd ee", "ff"]
        cpWER, hyp_min, ref_str = concat_perm_word_error_rate(spk_hypothesis=hyp, spk_reference=ref)
        calculated = 0
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6
        permuted_input_test(hyp, ref, calculated)

    @pytest.mark.unit
    def test_cpwer_ex1(self):
        hyp = ["aa bb c ff", "dd e ii jj kk", "hi"]
        ref = ["aa bb cc ff", "dd ee gg jj kk", "hh ii"]
        _ins, _del, _sub = 0, 1, 4
        cpWER, hyp_min, ref_str = concat_perm_word_error_rate(spk_hypothesis=hyp, spk_reference=ref)
        ref_word_count = word_count_ref(ref)
        calculated = calculate_wer_count(_ins, _del, _sub, ref_word_count)
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6
        permuted_input_test(hyp, ref, calculated)
        cpWER_perm, hyp_min_perm, ref_str = perm_concat_perm_word_error_rate(spk_hypothesis=hyp, spk_reference=ref)
        diff = torch.abs(torch.tensor(cpWER_perm - cpWER))
        assert diff <= 1e-6

    @pytest.mark.unit
    def test_cpwer_ex2(self):
        hyp = ["aa bb cc dd ii", "ff gg hh nn", "ee jj kk ll mm"]
        ref = ["aa bb cc dd", "ee ff gg hh ii", "jj kk ll mm nn"]
        _ins, _del, _sub = 2, 0, 2
        cpWER, hyp_min, ref_str = concat_perm_word_error_rate(spk_hypothesis=hyp, spk_reference=ref)
        ref_word_count = word_count_ref(ref)
        calculated = calculate_wer_count(_ins, _del, _sub, ref_word_count)
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6
        permuted_input_test(hyp, ref, calculated)
        cpWER_perm, hyp_min_perm, ref_str = perm_concat_perm_word_error_rate(spk_hypothesis=hyp, spk_reference=ref)
        diff = torch.abs(torch.tensor(cpWER_perm - cpWER))
        assert diff <= 1e-6

    @pytest.mark.unit
    def test_cpwer_ex3(self):
        hyp = ["aa bb cc", "dd ee", "gg", "hh ii", "jj kk"]
        ref = ["aa bb cc", "dd ee", "ff", "gg", "hh ii", "jj kk"]
        _ins, _del, _sub = 0, 1, 0
        cpWER, hyp_min, ref_str = concat_perm_word_error_rate(spk_hypothesis=hyp, spk_reference=ref)
        ref_word_count = word_count_ref(ref)
        calculated = calculate_wer_count(_ins, _del, _sub, ref_word_count)
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6
        permuted_input_test(hyp, ref, calculated)
        cpWER_perm, hyp_min_perm, ref_str = perm_concat_perm_word_error_rate(spk_hypothesis=hyp, spk_reference=ref)
        diff = torch.abs(torch.tensor(cpWER_perm - cpWER))
        assert diff <= 1e-6

    @pytest.mark.unit
    def test_cpwer_ex4(self):
        hyp = ["aa bb cc", "dd ee", "ff gg hh", "ii jj kk"]
        ref = ["aa bb cc", "dd ee", "ff gg hh ii jj kk"]
        _ins, _del, _sub = 0, 3, 0
        cpWER, hyp_min, ref_str = concat_perm_word_error_rate(spk_hypothesis=hyp, spk_reference=ref)
        ref_word_count = word_count_ref(ref)
        calculated = calculate_wer_count(_ins, _del, _sub, ref_word_count)
        diff = torch.abs(torch.tensor(calculated - cpWER))
        assert diff <= 1e-6
        permuted_input_test(hyp, ref, calculated)
