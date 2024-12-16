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
        lm = FastNGramLM.from_arpa(kenlm_model_path, vocab_size=1024)

    @pytest.mark.unit
    @pytest.mark.skipif(not KENLM_AVAILABLE, reason="KenLM is not available")
    def test_todo(self):
        arpa_lm_path = "/home/vbataev/code/nemo/check_beam_tdt/tdt_tune1_lmslurp.arpa.tmp.arpa"
        device = torch.device("cuda:0")
        _ = torch.tensor(0, device=device)

        lm = KenLMWrapper(arpa_lm_path)
        gpu_lm = FastNGramLM.from_arpa(arpa_lm_path, vocab_size=1024).to(device)

        with torch.no_grad():
            scores1, states1 = gpu_lm._compute_scores_batch_pytorch(states=gpu_lm.get_init_states(1, bos=True))
            scores2, states2 = gpu_lm._compute_scores_batch_triton(states=gpu_lm.get_init_states(1, bos=True))
        assert (states1 == states2).all()
        assert torch.allclose(scores1, scores2)

        batch_size = 2
        for _ in tqdm(range(10000)):
            start_state = random.randint(0, gpu_lm.num_states - 1)
            with torch.no_grad():
                scores1, states1 = gpu_lm._compute_scores_batch_pytorch(
                    states=torch.full([batch_size], fill_value=start_state, device=device, dtype=torch.int64)
                )
                scores2, states2 = gpu_lm._compute_scores_batch_cuda(
                    states=torch.full([batch_size], fill_value=start_state, device=device, dtype=torch.int64)
                )
                scores3, states3 = gpu_lm._compute_scores_batch_triton(
                    states=torch.full([batch_size], fill_value=start_state, device=device, dtype=torch.int64)
                )
            assert (states1 == states2).all()
            assert (states1 == states3).all()
            assert torch.allclose(scores1, scores2)
            assert torch.allclose(scores1, scores3)

        for bos in [False, True]:
            for i in range(1024):
                if abs(gpu_lm.compute_sentence_score([i], bos=bos) - lm.compute_sentence_score([i], bos=bos)) > 1e-4:
                    print(i, gpu_lm.compute_sentence_score([i], bos=bos), lm.compute_sentence_score([i], bos=bos))

        for sentence in [[25, 70, 12], [58, 41, 186, 293, 306, 999, 163, 264, 689, 683, 999]]:
            for bos in [True, False]:
                score1 = lm.compute_sentence_score(sentence, bos=False)
                score2 = gpu_lm.compute_sentence_score_cpu(sentence, bos=False)
                score3 = gpu_lm.compute_sentence_score(sentence, bos=False)
                assert abs(score1 - score2) < 1e-4
                assert abs(score1 - score3) < 1e-4

            # model_score = gpu_lm.compute_sentence_score(sentence, bos=bos, verbose=True)
            # lm_score = lm.compute_sentence_score(sentence, bos=bos)
            # model_score_batch = 0.0
            # states = gpu_lm.get_init_states(1, bos=bos)
            # for token in sentence:
            #     print(states)
            #     scores, new_states = compute_scores_batch(model, states)
            #     print(scores[0, token])
            #     model_score_batch += scores[0, token]
            #     states[:] = new_states[:, token]
            # print(bos, model_score, lm_score, model_score_batch)

    def test_autograd(self, test_data_dir):
        kenlm_model_path = Path(test_data_dir) / "asr/kenlm_ngram_lm/parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
        lm = FastNGramLM.from_arpa(kenlm_model_path, vocab_size=1024)
