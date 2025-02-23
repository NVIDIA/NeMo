# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from pathlib import Path

import pytest
import torch
from tqdm.auto import tqdm

from nemo.collections.asr.parts.submodules.ngram_lm import FastNGramLM, KenLMWrapper
from nemo.core.utils.optional_libs import KENLM_AVAILABLE, TRITON_AVAILABLE


class TestFastNGramLM:
    @pytest.mark.with_downloads
    @pytest.mark.unit
    def test_load(self, test_data_dir):
        kenlm_model_path = Path(test_data_dir) / "asr/kenlm_ngram_lm/parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
        _ = FastNGramLM.from_arpa(kenlm_model_path, vocab_size=1024)

    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.skipif(not KENLM_AVAILABLE, reason="KenLM is not available")
    def test_initial_states(self, test_data_dir):
        kenlm_model_path = Path(test_data_dir) / "asr/kenlm_ngram_lm/parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
        vocab_size = 1024
        device = torch.device("cpu")
        lm = FastNGramLM.from_arpa(kenlm_model_path, vocab_size=vocab_size, normalize_unk=False)
        kenlm_wrapper = KenLMWrapper(lm_path=kenlm_model_path, vocab_size=vocab_size)
        batch_size = 3
        for bos in [True, False]:
            init_states = lm.get_init_states(batch_size=batch_size, bos=bos)
            init_states_kenlm = kenlm_wrapper.get_init_states(batch_size=batch_size, bos=bos)
            scores_lm, _ = lm.advance(init_states)
            scores_ref, _ = kenlm_wrapper.advance(init_states_kenlm)
            assert torch.allclose(scores_lm, scores_ref.to(device))

    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_triton_vs_pytorch_random_states(self, test_data_dir):
        kenlm_model_path = Path(test_data_dir) / "asr/kenlm_ngram_lm/parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
        vocab_size = 1024
        lm = FastNGramLM.from_arpa(kenlm_model_path, vocab_size=vocab_size, normalize_unk=False)
        batch_size = 2
        device = torch.device("cuda")
        torch.manual_seed(777)
        lm = lm.to(device)
        for _ in tqdm(range(10000)):
            start_state = random.randint(0, lm.num_states - 1)
            with torch.no_grad():
                scores1, states1 = lm._advance_pytorch(
                    states=torch.full([batch_size], fill_value=start_state, device=device, dtype=torch.int64)
                )
                scores2, states2 = lm._advance_triton(
                    states=torch.full([batch_size], fill_value=start_state, device=device, dtype=torch.int64)
                )
            assert (states1 == states2).all()
            assert torch.allclose(scores1, scores2)

    @pytest.mark.unit
    @pytest.mark.skipif(not KENLM_AVAILABLE, reason="KenLM is not available")
    def test_sentences(self, test_data_dir):
        kenlm_model_path = Path(test_data_dir) / "asr/kenlm_ngram_lm/parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
        vocab_size = 1024
        device = torch.device("cpu")
        lm = FastNGramLM.from_arpa(kenlm_model_path, vocab_size=vocab_size, normalize_unk=False)
        kenlm_wrapper = KenLMWrapper(lm_path=kenlm_model_path, vocab_size=vocab_size)
        # TODO: make sentences
        for sentence in [[25, 70, 12], [58, 41, 186, 293, 306, 999, 163, 264, 689, 683, 999]]:
            for bos in [True, False]:
                score_lm = lm(torch.LongTensor([sentence]), bos=bos)
                score_ref = kenlm_wrapper.score_sentence(sentence, bos=bos)
                assert torch.allclose(score_lm[0], score_ref.to(device))
