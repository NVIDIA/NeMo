# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Beam search over a batch gives every source sequence the hypotheses and scores it gets alone."""

import pytest
import torch

from nemo.collections.asr.modules.transformer.transformer import TransformerDecoderNM
from nemo.collections.asr.modules.transformer.transformer_generators import (
    BeamSearchSequenceGenerator,
    BeamSearchSequenceGeneratorWithFusionModels,
)
from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.asr.parts.submodules.token_classifier import TokenClassifier

PAD, BOS, EOS, VOCAB, HIDDEN = 0, 1, 2, 12, 16


def _modules():
    torch.manual_seed(0)
    decoder = TransformerDecoderNM(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_layers=2,
        inner_size=24,
        num_attention_heads=2,
        max_sequence_length=40,
    ).eval()
    classifier = TokenClassifier(hidden_size=HIDDEN, num_classes=VOCAB, num_layers=1, log_softmax=True).eval()
    for parameter in list(decoder.parameters()) + list(classifier.parameters()):
        parameter.data.normal_(0, 0.5)
    return decoder, classifier


def _inputs(prompt_length):
    torch.manual_seed(1)
    batch, frames = 4, 7
    encoder_states = torch.randn(batch, frames, HIDDEN)
    encoder_mask = (torch.arange(frames)[None, :] < torch.tensor([7, 5, 6, 4])[:, None]).float()
    prompt = torch.full((batch, prompt_length), BOS, dtype=torch.long)
    if prompt_length > 1:
        prompt[:, 1:] = torch.randint(3, VOCAB, (batch, prompt_length - 1))
    return prompt, encoder_states, encoder_mask


def _assert_batched_matches_single(generator, prompt, encoder_states, encoder_mask):
    prefixes, scores, best, _ = generator(prompt, encoder_states, encoder_mask, return_beam_scores=True)
    for i in range(prompt.shape[0]):
        single_prefixes, single_scores, single_best, _ = generator(
            prompt[i : i + 1], encoder_states[i : i + 1], encoder_mask[i : i + 1], return_beam_scores=True
        )
        assert torch.equal(best[i], single_best[0]), i
        assert torch.equal(prefixes[i], single_prefixes[0]), i
        torch.testing.assert_close(scores[i], single_scores[0], atol=1e-5, rtol=1e-5)


class TestBeamSearchBatching:
    @pytest.mark.unit
    @pytest.mark.parametrize("prompt_length", [1, 3])
    def test_beam_search_batched_matches_single_sequence(self, prompt_length):
        decoder, classifier = _modules()
        generator = BeamSearchSequenceGenerator(
            decoder.embedding,
            decoder.decoder,
            classifier,
            beam_size=3,
            len_pen=0.5,
            bos=BOS,
            pad=PAD,
            eos=EOS,
            max_delta_length=4,
        )
        _assert_batched_matches_single(generator, *_inputs(prompt_length))

    @pytest.mark.unit
    def test_beam_search_with_fusion_models_batched_matches_single_sequence(self):
        decoder, classifier = _modules()
        generator = BeamSearchSequenceGeneratorWithFusionModels(
            decoder.embedding,
            decoder.decoder,
            classifier,
            fusion_models=[NGramGPULanguageModel.dummy_unigram_lm(vocab_size=VOCAB)],
            fusion_models_alpha=[0.3],
            beam_size=3,
            len_pen=0.5,
            bos=BOS,
            pad=PAD,
            eos=EOS,
            max_delta_length=4,
        )
        _assert_batched_matches_single(generator, *_inputs(3))
