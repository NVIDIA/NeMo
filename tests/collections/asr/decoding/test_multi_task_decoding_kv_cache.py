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

"""The sequence generators give the same hypotheses over the decoder's key-value cache as over cached inputs."""

import functools
from unittest.mock import Mock

import pytest
import torch

from nemo.collections.asr.modules.transformer.transformer import TransformerDecoderNM
from nemo.collections.asr.modules.transformer.transformer_generators import (
    BeamSearchSequenceGenerator,
    BeamSearchSequenceGeneratorWithFusionModels,
    GreedySequenceGenerator,
)
from nemo.collections.asr.modules.transformer.transformer_modules import MultiHeadAttention, RMSNorm
from nemo.collections.asr.parts.submodules.multitask_beam_decoding import AEDBeamInferConfig, TransformerAEDBeamInfer
from nemo.collections.asr.parts.submodules.multitask_decoding import MultiTaskDecoding, MultiTaskDecodingConfig
from nemo.collections.asr.parts.submodules.multitask_greedy_decoding import (
    AEDGreedyInferConfig,
    TransformerAEDGreedyInfer,
)
from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.asr.parts.submodules.token_classifier import TokenClassifier

PAD, BOS, EOS, VOCAB, HIDDEN = 0, 1, 2, 12, 16
TOLERANCE = dict(atol=1e-5, rtol=1e-5)


@pytest.fixture()
def modules():
    torch.manual_seed(0)
    decoder = TransformerDecoderNM(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_layers=2,
        inner_size=24,
        num_attention_heads=2,
        max_sequence_length=40,
        num_token_types=0,
        pre_ln=True,
        norm=functools.partial(RMSNorm, eps=1e-6),
        self_attn=functools.partial(MultiHeadAttention, use_bias=False, rotary_base=10000.0),
    ).eval()
    classifier = TokenClassifier(hidden_size=HIDDEN, num_classes=VOCAB, num_layers=1, log_softmax=True).eval()
    for parameter in list(decoder.parameters()) + list(classifier.parameters()):
        parameter.data.normal_(0, 0.5)
    # Trained models never emit the pad token before the end of a hypothesis. A random model would, and the
    # cache then masks the pad position as the full pass does, while the memory path attends to it.
    classifier.mlp.layer0.bias.data[PAD] = -30.0
    return decoder, classifier


@pytest.fixture()
def inputs():
    torch.manual_seed(1)
    batch, frames = 3, 7
    prompt = torch.tensor([[BOS, 4], [BOS, 7], [BOS, 3]])
    encoder_states = torch.randn(batch, frames, HIDDEN)
    encoder_mask = (torch.arange(frames)[None, :] < torch.tensor([7, 5, 6])[:, None]).float()
    return prompt, encoder_states, encoder_mask


@pytest.fixture()
def tokenizer():
    tok = Mock()
    tok.pad, tok.bos, tok.eos = PAD, BOS, EOS
    return tok


def _generators(cls, decoder, classifier, **kwargs):
    return [
        cls(decoder.embedding, decoder.decoder, classifier, bos=BOS, pad=PAD, eos=EOS, use_kv_cache=flag, **kwargs)
        for flag in (False, True)
    ]


class TestSequenceGeneratorsKeyValueCache:
    @pytest.mark.unit
    def test_greedy(self, modules, inputs):
        default, cached = _generators(
            GreedySequenceGenerator,
            *modules,
            max_delta_length=5,
            return_xattn_scores=True,
            preserve_step_confidence=True,
        )
        best, _, confidence, xatt = default(*inputs)
        cached_best, _, cached_confidence, cached_xatt = cached(*inputs)
        assert torch.equal(best, cached_best)
        torch.testing.assert_close(confidence, cached_confidence, **TOLERANCE)
        for layer in range(len(xatt)):
            torch.testing.assert_close(xatt[layer], cached_xatt[layer], **TOLERANCE)

    @pytest.mark.unit
    def test_temperature_sampling(self, modules, inputs):
        default, cached = _generators(
            GreedySequenceGenerator, *modules, max_delta_length=5, temperature=2.0, n_samples=2
        )
        torch.manual_seed(3)
        best, samples, _, _ = default(*inputs)
        torch.manual_seed(3)
        cached_best, cached_samples, _, _ = cached(*inputs)
        assert torch.equal(best, cached_best)
        assert all(torch.equal(a, b) for a, b in zip(samples, cached_samples))

    @pytest.mark.unit
    def test_beam(self, modules, inputs):
        default, cached = _generators(
            BeamSearchSequenceGenerator,
            *modules,
            beam_size=3,
            len_pen=0.5,
            max_delta_length=5,
            return_xattn_scores=True,
        )
        prefixes, scores, best, xatt = default(*inputs, return_beam_scores=True)
        cached_prefixes, cached_scores, cached_best, cached_xatt = cached(*inputs, return_beam_scores=True)
        assert torch.equal(best, cached_best)
        for i in range(len(prefixes)):
            assert torch.equal(prefixes[i], cached_prefixes[i])
            torch.testing.assert_close(scores[i], cached_scores[i], **TOLERANCE)
        for layer in range(len(xatt)):
            torch.testing.assert_close(xatt[layer], cached_xatt[layer], **TOLERANCE)

    @pytest.mark.unit
    def test_beam_with_fusion_models(self, modules, inputs):
        default, cached = _generators(
            BeamSearchSequenceGeneratorWithFusionModels,
            *modules,
            fusion_models=[NGramGPULanguageModel.dummy_unigram_lm(vocab_size=VOCAB)],
            fusion_models_alpha=[0.3],
            beam_size=3,
            len_pen=0.5,
            max_delta_length=5,
        )
        prefixes, scores, best, _ = default(*inputs, return_beam_scores=True)
        cached_prefixes, cached_scores, cached_best, _ = cached(*inputs, return_beam_scores=True)
        assert torch.equal(best, cached_best)
        for i in range(len(prefixes)):
            assert torch.equal(prefixes[i], cached_prefixes[i])
            torch.testing.assert_close(scores[i], cached_scores[i], **TOLERANCE)

    @pytest.mark.unit
    def test_aed_infer_and_configuration(self, modules, inputs, tokenizer):
        decoder, classifier = modules
        prompt, encoder_states, encoder_mask = inputs
        results = {}
        for flag in (False, True):
            greedy = TransformerAEDGreedyInfer(
                decoder, classifier, tokenizer, max_generation_delta=5, use_kv_cache=flag
            )
            beam = TransformerAEDBeamInfer(
                decoder, classifier, tokenizer, beam_size=3, max_generation_delta=5, use_kv_cache=flag
            )
            assert greedy.greedy_search.use_kv_cache is flag and beam.beam_search.use_kv_cache is flag
            results[flag] = [
                [
                    hyp.y_sequence
                    for hyp in infer(
                        encoder_hidden_states=encoder_states, encoder_input_mask=encoder_mask, decoder_input_ids=prompt
                    )[0]
                ]
                for infer in (greedy, beam)
            ]
        for default, cached in zip(results[False], results[True]):
            assert all(torch.equal(a, b) for a, b in zip(default, cached))

        config = MultiTaskDecodingConfig(
            strategy="greedy",
            greedy=AEDGreedyInferConfig(use_kv_cache=True),
            beam=AEDBeamInferConfig(beam_size=2, use_kv_cache=True),
        )
        decoding = MultiTaskDecoding(config, decoder, classifier, tokenizer)
        assert decoding.decoding.greedy_search.use_kv_cache is True
        decoding.change_strategy("beam")
        assert decoding.decoding.beam_search.use_kv_cache is True
        default = MultiTaskDecoding(MultiTaskDecodingConfig(strategy="beam"), decoder, classifier, tokenizer)
        assert default.decoding.beam_search.use_kv_cache is False
