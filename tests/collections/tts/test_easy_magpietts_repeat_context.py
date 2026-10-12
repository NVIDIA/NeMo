# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conditioning-channel tests without downloading codec or validation models."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel, TrainingMode
from nemo.collections.tts.modules.magpietts_modules import LocalTransformerType


class _ChannelHarness(nn.Module):
    """Exercise production channel assembly with a tiny decoder and embeddings."""

    prepare_context_tensors = EasyMagpieTTSInferenceModel.prepare_context_tensors
    join_embeddings_temporally = EasyMagpieTTSInferenceModel.join_embeddings_temporally
    stack_codes = EasyMagpieTTSInferenceModel.stack_codes
    slice_sequence_embeddings = EasyMagpieTTSModel.slice_sequence_embeddings
    _repeat_context_embeddings = staticmethod(EasyMagpieTTSInferenceModel._repeat_context_embeddings)
    streaming_init = EasyMagpieTTSInferenceModel.streaming_init
    streaming_prefill = EasyMagpieTTSInferenceModel.streaming_prefill
    _prepare_streaming_input = EasyMagpieTTSInferenceModel._prepare_streaming_input
    process_batch = EasyMagpieTTSModel.process_batch
    _is_text_only_batch = staticmethod(EasyMagpieTTSModel._is_text_only_batch)
    _process_text_only_batch = EasyMagpieTTSModel._process_text_only_batch
    prepare_text_channel_embeddings = EasyMagpieTTSModel.prepare_text_channel_embeddings
    compute_loss = EasyMagpieTTSModel.compute_loss
    compute_phoneme_loss = EasyMagpieTTSModel.compute_phoneme_loss

    def __init__(self, repeat_context=False, device='cpu'):
        super().__init__()
        cfg = {'embedding_dim': 4, 'use_multiturn_dataset': False}
        if repeat_context is not None:
            cfg['repeat_context'] = repeat_context
        self.cfg = OmegaConf.create(cfg)
        self.repeat_context = self.cfg.get('repeat_context', False)
        self.training_modes = [TrainingMode('streaming', 1, 2, 0)]
        self.default_inference_mode = self.training_modes[0].name
        self.mode_name_to_mode = {self.default_inference_mode: self.training_modes[0]}
        self.task_embedding = None
        self.use_speaker_encoder = True
        self.train_shuffle_context_embedding_prob = 0.0
        self.disable_cas_for_context_text = True
        self._codec_converter = None
        self.num_audio_codebooks = self.data_num_audio_codebooks = self.frame_stacking_factor = 1
        self.context_audio_bos_id = 10
        self.context_audio_eos_id = 11
        self.audio_bos_id = 12
        self.audio_eos_id = 13
        self.audio_user_speaking_id = 14
        self.pad_id = 0
        self.eos_id = 15
        self.cfg_unk_token_id = 16
        self.tokenizer = SimpleNamespace(pad=0)
        self.phoneme_tokenizer = None
        self.codec_sil_codes = torch.tensor([1], device=device)
        self.text_lookup = nn.Embedding(32, 4)
        self.audio_lookup = nn.Embedding(32, 4)
        with torch.no_grad():
            self.text_lookup.weight.copy_(torch.arange(128).reshape(32, 4) / 128)
            self.audio_lookup.weight.copy_(torch.arange(128).reshape(32, 4) / 256)
        self.audio_out_projection = nn.Identity()
        self.final_proj = nn.Linear(4, 8)
        self.phoneme_final_proj = nn.Linear(4, 32)
        self.phoneme_stacking_factor = 1
        self.phoneme_vocab_size = 32
        self.num_all_tokens_per_codebook = 8
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.dropout_text_input_prob = 0.0
        self.cfg_unconditional_prob = 0.0
        self.feature_masking = None
        self.phoneme_corruption_type = 'repeat_skip_unk'
        self.parallel_codebook_loss_scale = self.phoneme_loss_weight = 1.0
        self.local_transformer_type = LocalTransformerType.NO_LT
        self.forward_inputs = []
        self.to(device)

    def embed_text_tokens(self, tokens, **kwargs):
        return self.text_lookup(tokens.long())

    def embed_audio_tokens(self, codes):
        return self.audio_lookup(codes[:, 0].long())

    def encode_context_audio_embeddings(self, context_audio_embedded, context_audio_lens):
        return context_audio_embedded * 2

    def forward(self, inputs_embeds, **kwargs):
        self.forward_inputs.append(inputs_embeds)
        return SimpleNamespace(last_hidden_state=inputs_embeds, past_key_values='test-cache')

    def prepare_audio_channel_embeddings(self, audio_codes, audio_codes_lens, delay, **kwargs):
        embedding, lengths = self.join_embeddings_temporally(
            [
                self.audio_lookup.weight.new_zeros(delay.numel(), int(delay.max()), 4),
                self.embed_audio_tokens(audio_codes),
            ],
            [delay, audio_codes_lens],
        )
        return embedding, lengths, audio_codes, audio_codes_lens, None

    def prepare_phoneme_channel_embeddings(self, phoneme_tokens, phoneme_tokens_lens, delay, **kwargs):
        # Keep losses real while using short teacher-forced phoneme inputs.
        embedding, lengths = self.join_embeddings_temporally(
            [
                self.text_lookup.weight.new_zeros(delay.numel(), int(delay.max()), 4),
                self.embed_text_tokens(phoneme_tokens),
            ],
            [delay, phoneme_tokens_lens],
        )
        targets = torch.cat([phoneme_tokens[:, :1], phoneme_tokens], dim=1).unsqueeze(1)
        return embedding, lengths, targets, phoneme_tokens_lens + 1, targets, None


def _batch(device='cpu'):
    return dict(
        text=torch.tensor([[2, 3, 4], [5, 6, 0]], device=device),
        text_lens=torch.tensor([3, 2], device=device),
        context_text_tokens=torch.tensor([[7, 8], [9, 0]], device=device),
        context_text_tokens_lens=torch.tensor([2, 1], device=device),
        audio_codes=torch.ones(2, 1, 3, dtype=torch.long, device=device),
        audio_codes_lens=torch.tensor([3, 3], device=device),
        context_audio_codes=torch.tensor([[[2, 3]], [[4, 0]]], device=device),
        context_audio_codes_lens=torch.tensor([2, 1], device=device),
        mode='val',
    )


def _init(model, use_cfg=False):
    batch = _batch(model.text_lookup.weight.device)
    return model.streaming_init(
        **{key: value for key, value in batch.items() if key.startswith('context_')},
        use_cfg=use_cfg,
        use_inference_mode=False,
    )


@pytest.mark.unit
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_repeat_ignores_padding_and_preserves_gradient(dtype):
    context = torch.tensor([[[1.0], [2.0], [999.0]], [[3.0], [4.0], [5.0]]], dtype=dtype, requires_grad=True)
    actual = EasyMagpieTTSInferenceModel._repeat_context_embeddings(
        context, torch.tensor([2, 3]), torch.arange(7), torch.tensor([5, 7])
    )
    expected = torch.tensor([[[1], [2], [1], [2], [1], [0], [0]], [[3], [4], [5], [3], [4], [5], [3]]], dtype=dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    torch.testing.assert_close(context.grad[..., 0], torch.tensor([[3, 2, 0], [3, 2, 2]], dtype=dtype))


@pytest.mark.unit
def test_empty_context_is_zero():
    repeat = EasyMagpieTTSInferenceModel._repeat_context_embeddings
    assert not repeat(torch.empty(2, 0, 4), torch.zeros(2, dtype=torch.long), torch.arange(3)).any()
    assert not repeat(torch.ones(2, 1, 4), torch.zeros(2, dtype=torch.long), torch.arange(3)).any()


@pytest.mark.unit
@pytest.mark.parametrize('repeat', [None, False, True])
@pytest.mark.parametrize('text_only', [False, True])
def test_training_context_channel_and_default(repeat, text_only):
    model = _ChannelHarness(repeat).eval()
    batch = _batch()
    if text_only:
        batch['context_text_tokens'][1, 1] = 10
        batch['context_text_tokens_lens'][1] = 2
        model.phoneme_tokenizer = SimpleNamespace(eos_token_id=17, pad=0)
        context = model.embed_text_tokens(batch['context_text_tokens'])
        context_lens = batch['context_text_tokens_lens']
        batch.update(
            task=['text_only', 'text_only'], phoneme_tokens=batch['text'], phoneme_tokens_lens=batch['text_lens']
        )
    else:
        context, context_lens, _, _ = model.prepare_context_tensors(
            **{key: value for key, value in batch.items() if key.startswith('context_')}
        )

    # Subtract an otherwise identical run with zero context to isolate its channel.
    control = _ChannelHarness(False).eval()
    control.load_state_dict(model.state_dict())
    control.phoneme_tokenizer = model.phoneme_tokenizer
    if text_only:
        original_embed = control.embed_text_tokens
        control.embed_text_tokens = lambda tokens, **kw: (
            torch.zeros_like(original_embed(tokens))
            if tokens is batch['context_text_tokens']
            else original_embed(tokens)
        )
    else:
        control.prepare_context_tensors = lambda **kw: (torch.zeros_like(context), context_lens, None, None)
    model.process_batch(**batch)
    control.process_batch(**batch)
    contribution = model.forward_inputs[-1] - control.forward_inputs[-1]
    for item, length in enumerate(context_lens.tolist()):
        torch.testing.assert_close(contribution[item, :length], context[item, :length])
        if repeat:
            torch.testing.assert_close(contribution[item, length], context[item, 0])
        else:
            assert not contribution[item, length:].any()
    assert torch.isfinite(model.forward_inputs[-1]).all()


@pytest.mark.unit
@pytest.mark.parametrize('repeat', [False, True])
@pytest.mark.parametrize('use_cfg', [False, True])
def test_streaming_context_phase_and_cfg(repeat, use_cfg):
    model = _ChannelHarness(repeat).eval()
    state = _init(model, use_cfg)
    prefix_input = model.forward_inputs[-1]
    torch.testing.assert_close(prefix_input[:2], state.full_context_embedding[:, : state.cache_seq_len])
    actual, needs_context, _, _ = model._prepare_streaming_input(state, None, False)
    # Longer item is still processing its prefix: no duplicate conditioning.
    assert needs_context.tolist() == [True, False]
    torch.testing.assert_close(actual[0, 0], state.full_context_embedding[0, state.cache_seq_len])
    expected_short = state.full_context_embedding[1, 0] if repeat else torch.zeros(4)
    torch.testing.assert_close(actual[1, 0], expected_short)
    if use_cfg:
        dummy = state.config.dummy_context_embedding_unconditional[0, 0]
        torch.testing.assert_close(actual[2, 0], dummy)
        torch.testing.assert_close(actual[3, 0], dummy if repeat else torch.zeros(4))


@pytest.mark.unit
@pytest.mark.parametrize('use_cfg', [False, True])
def test_prefill_chunking_and_next_turn_continue_same_context(use_cfg):
    model = _ChannelHarness(True).eval()
    state = _init(model, use_cfg)
    # Finish both prefixes before a user-speech prefill.
    state.cache_seq_len = int(state.full_context_lens.max())
    state.context_position = state.full_context_lens.clone()
    start = state.cache_seq_len
    tokens = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
    model.streaming_prefill(state, tokens, use_inference_mode=False)
    whole = model.forward_inputs[-1].clone()
    split_state = _init(model, use_cfg)
    split_state.cache_seq_len = start
    split_state.context_position = split_state.full_context_lens.clone()
    pieces = []
    for piece in tokens.split(2, dim=1):
        model.streaming_prefill(split_state, piece, use_inference_mode=False)
        pieces.append(model.forward_inputs[-1].clone())
    torch.testing.assert_close(torch.cat(pieces, dim=1), whole, rtol=0, atol=0)

    # A new turn resets turn counters, but not the context channel's absolute phase.
    state.turn_text_tokens_seen.zero_()
    state.text_tokens_seen.zero_()
    state.text_finished.fill_(True)
    actual, _, _, _ = model._prepare_streaming_input(state, None, False)
    for item, length in enumerate(state.full_context_lens.tolist()):
        torch.testing.assert_close(actual[item, 0], state.full_context_embedding[item, state.cache_seq_len % length])
    if use_cfg:
        torch.testing.assert_close(actual[2:], state.config.dummy_context_embedding_unconditional.expand(2, 1, -1))


@pytest.mark.unit
def test_zero_shot_blocker_is_applied_before_repetition(monkeypatch):
    model = _ChannelHarness(True).train()
    model.train_shuffle_context_embedding_prob = 1.0
    monkeypatch.setattr('nemo.collections.tts.models.easy_magpietts_inference.random.randint', lambda *args: 1)
    batch = _batch()
    context, lengths, codes, _ = model.prepare_context_tensors(
        **{key: value for key, value in batch.items() if key.startswith('context_')}
    )
    raw_shuffled = model.embed_audio_tokens(codes).roll(1, dims=0)
    # The blocker bypasses the speaker encoder and shuffles the raw reference.
    torch.testing.assert_close(context[:, :2], raw_shuffled[:, :2])
    repeated = model._repeat_context_embeddings(context, lengths, lengths[:1])
    torch.testing.assert_close(repeated[0, 0], raw_shuffled[0, 0])
    model.process_batch(**batch)
    expected = raw_shuffled[0, 0] + model.embed_text_tokens(batch['text'])[0, 0]
    torch.testing.assert_close(model.forward_inputs[-1][0, int(lengths[0])], expected)


@pytest.mark.unit
def test_cfg_dropout_repeats_only_unconditional_context():
    model = _ChannelHarness(True).train()
    model.cfg_unconditional_prob = 1.0
    batch = _batch()
    batch['mode'] = 'train'
    model.process_batch(**batch)
    actual = model.forward_inputs[-1]
    dummy = model.embed_text_tokens(torch.tensor([[model.cfg_unk_token_id]]))[0, 0]
    # Context + dropped text only, before the audio stream starts.
    torch.testing.assert_close(actual[0, :6], dummy.expand(6, -1))


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')
def test_cuda_training_backpropagates_through_repeated_reference():
    model = _ChannelHarness(True, 'cuda').train()
    output = model.process_batch(**_batch('cuda'))
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert model.audio_lookup.weight.grad is not None
    assert model.audio_lookup.weight.grad[10].abs().sum() > 0  # repeated reference BOS
    assert model.text_lookup.weight.grad[7].abs().sum() > 0  # repeated context text
