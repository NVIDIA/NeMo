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

from nemo.collections.asr.parts.submodules.ngram_lm import FastNGramLM, KenLMBatchedWrapper
from nemo.core.utils.optional_libs import KENLM_AVAILABLE, TRITON_AVAILABLE
from torch.nn.utils.rnn import pad_sequence

DEVICES = [torch.device("cpu")]

if torch.cuda.is_available():
    DEVICES.append('cuda')


class TestFastNGramLM:
    @pytest.mark.with_downloads
    @pytest.mark.unit
    def test_load(self, test_data_dir):
        kenlm_model_path = Path(test_data_dir) / "asr/kenlm_ngram_lm/parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
        _ = FastNGramLM.from_arpa(kenlm_model_path, vocab_size=1024)

    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.skipif(not KENLM_AVAILABLE, reason="KenLM is not available")
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("bos", [True, False])
    def test_initial_states(self, test_data_dir, bos: bool, batch_size: int, device: torch.device):
        kenlm_model_path = Path(test_data_dir) / "asr/kenlm_ngram_lm/parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
        vocab_size = 1024
        n_gpu_lm = FastNGramLM.from_arpa(kenlm_model_path, vocab_size=vocab_size, normalize_unk=False).to(device)
        kenlm_wrapper = KenLMBatchedWrapper(lm_path=kenlm_model_path, vocab_size=vocab_size)

        init_states = n_gpu_lm.get_init_states(batch_size=batch_size, bos=bos)
        init_states_kenlm = kenlm_wrapper.get_init_states(batch_size=batch_size, bos=bos)
        scores_lm, _ = n_gpu_lm.advance(init_states)
        scores_ref, _ = kenlm_wrapper.advance(init_states_kenlm)
        assert torch.allclose(scores_lm, scores_ref.to(device))

    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_triton_vs_pytorch_random_states(self, test_data_dir, batch_size=2, num_iterations=100):
        """Randomly initializes the states and compares the scores from Triton and PyTorch implementations."""
        torch.manual_seed(777)
        device = torch.device("cuda")
        vocab_size = 1024
        kenlm_model_path = Path(test_data_dir) / "asr/kenlm_ngram_lm/parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
        n_gpu_lm = FastNGramLM.from_arpa(kenlm_model_path, vocab_size=vocab_size, normalize_unk=False).to(device)
        for _ in tqdm(range(num_iterations)):
            start_state = random.randint(0, n_gpu_lm.num_states - 1)
            with torch.no_grad():
                scores1, states1 = n_gpu_lm._advance_pytorch(
                    states=torch.full([batch_size], fill_value=start_state, device=device, dtype=torch.int64)
                )
                scores2, states2 = n_gpu_lm._advance_triton(
                    states=torch.full([batch_size], fill_value=start_state, device=device, dtype=torch.int64)
                )
            assert (states1 == states2).all()
            assert torch.allclose(scores1, scores2)

    @pytest.mark.unit
    @pytest.mark.skipif(not KENLM_AVAILABLE, reason="KenLM is not available")
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("bos", [True, False])
    def test_sentences(self, test_data_dir, bos: bool, device: torch.device):
        kenlm_model_path = Path(test_data_dir) / "asr/kenlm_ngram_lm/parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
        vocab_size = 1024
        n_gpu_lm = FastNGramLM.from_arpa(kenlm_model_path, vocab_size=vocab_size, normalize_unk=False).to(device)
        kenlm_wrapper = KenLMBatchedWrapper(lm_path=kenlm_model_path, vocab_size=vocab_size)
        sentences = [
            [25, 70, 12],
            [58, 41, 186, 293, 306, 999, 163, 264, 689, 683, 999],
        ]
        scores_ref = kenlm_wrapper.score_sentences(sentences, bos=bos).to(device)
        scores_lm_not_batched = torch.zeros_like(scores_ref)
        for i, sentence in enumerate(sentences):
            current_score_lm = n_gpu_lm(torch.LongTensor([sentence]).to(device), bos=bos)
            scores_lm_not_batched[i] += current_score_lm.squeeze()
        assert torch.allclose(scores_lm_not_batched, scores_ref)

        scores_lm_batched = n_gpu_lm(
            labels=pad_sequence([torch.LongTensor(sentence) for sentence in sentences], batch_first=True).to(device),
            labels_lengths=torch.LongTensor([len(sentence) for sentence in sentences], device=device),
            bos=bos,
        )
        assert torch.allclose(scores_lm_batched, scores_ref)
