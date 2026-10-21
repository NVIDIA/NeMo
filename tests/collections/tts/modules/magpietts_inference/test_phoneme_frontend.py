# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for independent text decoding and per-turn forced IPA."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from examples.tts.magpietts_inference import create_argument_parser
from omegaconf import OmegaConf

from nemo.collections.tts.modules.magpietts_inference.inference import (
    EasyMagpieInferenceConfig,
    EasyMagpieInferenceRunner,
    EasyMagpieMultiturnUserAudioInferenceConfig,
    EasyMagpieMultiturnUserAudioInferenceRunner,
    _frontend_phoneme_tokens,
)
from nemo.collections.tts.modules.magpietts_inference.phoneme_frontend import TextOnlyPhonemeFrontend
from nemo.collections.tts.modules.magpietts_inference.utils import _build_easy_magpie_config, _select_runner_cls

CLI_PATHS = ['--codecmodel_path', '/codec.nemo', '--datasets_json_path', '/datasets.json', '--out_dir', '/output']


class TinyFrontend(torch.nn.Module):
    def __init__(self, delay=8, emit_eos=True, decoded='wˈʌn'):
        super().__init__()
        self.cfg = OmegaConf.create({'use_multiturn_dataset': True, 'text_only_cas_chunk_size': 256})
        self.device = torch.device('cpu')
        self.phoneme_tokenizer = SimpleNamespace(bos_token_id=80, eos_token_id=81, decode=Mock(return_value=decoded))
        self.tokenizer = SimpleNamespace(encode=Mock(return_value=[11, 12]))
        self.text_conditioning_tokenizer_name = 'text'
        self.phoneme_stacking_factor = 1
        self.training_modes = [
            SimpleNamespace(
                text_input_mode='streaming', streaming_phonemes_delay=delay, mode_idx=None, name=f'streaming_ph{delay}'
            )
        ]
        self.task_embedding = None
        self.disable_cas_for_context_text = False
        self.repeat_context = False
        self.eos_id, self.pad_id = 13, 0
        self.forwards, self.phoneme_inputs = [], []
        self.emit_eos = emit_eos

    def embed_text_tokens(self, ids, **kwargs):
        return torch.ones(ids.size(0), ids.size(1), 4)

    def embed_phoneme_tokens(self, ids):
        self.phoneme_inputs.append(ids.clone())
        return torch.zeros(ids.size(0), ids.size(2), 4)

    def forward(self, **kwargs):
        self.forwards.append(kwargs)
        return SimpleNamespace(last_hidden_state=kwargs['inputs_embeds'], past_key_values='cache')

    def phoneme_final_proj(self, hidden):
        logits = torch.full((1, 1, 82), -100.0)
        logits[:, :, 81 if self.emit_eos and len(self.forwards) > 1 else 7] = 100.0
        return logits


def frontend(monkeypatch, tmp_path, **kwargs):
    model = TinyFrontend(**kwargs)
    checkpoint = tmp_path / 'model.nemo'
    checkpoint.write_bytes(b'fixture')
    restore = Mock(side_effect=[OmegaConf.create({}), model])
    monkeypatch.setattr(
        'nemo.collections.tts.modules.magpietts_inference.phoneme_frontend.EasyMagpieTTSInferenceModel.restore_from',
        restore,
    )
    return TextOnlyPhonemeFrontend(str(checkpoint), 'cpu', max_steps=2), model


@pytest.mark.unit
def test_frontend_raw_text_delay_eos_cache_and_tf32_restoration(monkeypatch, tmp_path):
    decoder, model = frontend(monkeypatch, tmp_path)
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    output = decoder.generate('$1', 'en')
    assert output.ipa == 'wˈʌn' and output.token_ids == (7,)
    assert output.decoder_steps == 2
    assert [call.args[0] for call in model.tokenizer.encode.call_args_list] == ['[EN][TEXT_ONLY]', '$1']
    # Two context tokens, delay eight: BOS is at absolute position ten.
    assert model.forwards[0]['inputs_embeds'].shape == (1, 11, 4)
    assert model.forwards[1]['cache_position'].tolist() == [11]
    assert model.phoneme_inputs[0].tolist() == [[[80]]]
    assert model.phoneme_inputs[1].tolist() == [[[7]]]
    assert model.forwards[1]['past_key_values'] == 'cache'
    assert model.forwards[0]['context_prefix_lens'].tolist() == [2]
    model.phoneme_tokenizer.decode.assert_called_once_with([7])
    assert decoder.generate('$1', 'en') is output
    assert len(model.forwards) == 2
    assert before == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


@pytest.mark.unit
@pytest.mark.parametrize('kwargs,match', [({'emit_eos': False}, 'did not reach EOS'), ({'decoded': ''}, 'empty IPA')])
def test_frontend_rejects_incomplete_output_without_reference_fallback(monkeypatch, tmp_path, kwargs, match):
    decoder, _ = frontend(monkeypatch, tmp_path, **kwargs)
    with pytest.raises(RuntimeError, match=match):
        decoder.generate('$1', 'en')
    assert not decoder._cache


@pytest.mark.unit
def test_frontend_rejects_wrong_checkpoint_delay(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match='phoneme delay 8'):
        frontend(monkeypatch, tmp_path, delay=3)


@pytest.mark.unit
def test_forced_ipa_retokenizes_and_replaces_previous_turn_before_warmup():
    model = SimpleNamespace(
        device='cpu',
        cfg={'use_multiturn_dataset': True},
        phoneme_stacking_factor=1,
        phoneme_tokenizer=SimpleNamespace(
            bos_token_id=100, eos_token_id=101, pad=0, encode=Mock(side_effect=[[4, 5], [6]])
        ),
    )
    model.stack_codes = lambda codes, lengths, *args: (codes, lengths)
    model.embed_phoneme_tokens = lambda codes: codes.transpose(1, 2).float()
    runner = object.__new__(EasyMagpieMultiturnUserAudioInferenceRunner)
    runner.model = model
    state = SimpleNamespace(gt_phoneme_embeddings=None, gt_phoneme_lens=None)
    assert runner._set_turn_phoneme_inputs(state, 'wˈʌn') == [100, 4, 5, 101]
    assert state.gt_phoneme_embeddings[0, :, 0].tolist() == [100, 4, 5, 101]
    assert state.gt_phoneme_lens.tolist() == [4]
    assert runner._set_turn_phoneme_inputs(state, 'tˈuː') == [100, 6, 101]
    assert state.gt_phoneme_embeddings[0, :, 0].tolist() == [100, 6, 101]
    assert state.gt_phoneme_lens.tolist() == [3]
    assert runner._set_turn_phoneme_inputs(state, 'unused IPA', token_ids=[100, 9, 10, 101]) == [100, 9, 10, 101]
    assert state.gt_phoneme_embeddings[0, :, 0].tolist() == [100, 9, 10, 101]
    assert model.phoneme_tokenizer.encode.call_count == 2


@pytest.mark.unit
def test_disabled_frontend_preserves_multiturn_identifier():
    config = EasyMagpieMultiturnUserAudioInferenceConfig()
    assert config.phoneme_frontend_nemo_file is None
    assert config.build_identifier().endswith('_MTUserAudio_MaxTurns6')
    config.phoneme_frontend_nemo_file = '/model/frontend.nemo'
    assert '_FrontendD8_' in config.build_identifier()


@pytest.mark.unit
@pytest.mark.parametrize(
    'options',
    [
        ['--easy_magpie_inference_mode', mode, '--phoneme_input_type', 'predicted']
        for mode in ('single_turn', 'multiturn_user_audio')
    ],
)
def test_cli_rejects_frontend_without_forced_phonemes(options):
    args = create_argument_parser().parse_args(
        CLI_PATHS + ['--phoneme_frontend_nemo_file', '/frontend.nemo'] + options
    )
    with pytest.raises(ValueError, match='phoneme_input_type gt'):
        _build_easy_magpie_config(args)


@pytest.mark.unit
def test_cli_builds_frontend_config_and_rejects_encoder_decoder():
    args = create_argument_parser().parse_args(
        CLI_PATHS
        + [
            '--model_type',
            'easy_magpie',
            '--easy_magpie_inference_mode',
            'multiturn_user_audio',
            '--phoneme_input_type',
            'gt',
            '--phoneme_frontend_nemo_file',
            '/frontend.nemo',
        ]
    )
    config = _build_easy_magpie_config(args)
    assert config.phoneme_frontend_nemo_file == '/frontend.nemo'
    assert config.phoneme_frontend_expected_delay == 8
    assert _select_runner_cls(args) is EasyMagpieMultiturnUserAudioInferenceRunner
    args.model_type = 'magpie'
    with pytest.raises(ValueError, match='easy_magpie'):
        _select_runner_cls(args)


@pytest.mark.unit
def test_cli_accepts_single_turn_frontend():
    args = create_argument_parser().parse_args(
        CLI_PATHS
        + [
            '--model_type',
            'easy_magpie',
            '--phoneme_input_type',
            'gt',
            '--phoneme_frontend_nemo_file',
            '/frontend.nemo',
        ]
    )
    config = _build_easy_magpie_config(args)
    assert type(config) is EasyMagpieInferenceConfig
    assert config.phoneme_frontend_nemo_file == '/frontend.nemo'
    assert '_FrontendD8_' in config.build_identifier()
    assert _select_runner_cls(args) is EasyMagpieInferenceRunner


@pytest.mark.unit
def test_single_turn_frontend_overwrites_reference_ipa_with_tts_tokenizer():
    runner = object.__new__(EasyMagpieInferenceRunner)
    runner.model = SimpleNamespace(
        phoneme_tokenizer=SimpleNamespace(
            bos_token_id=100, eos_token_id=101, pad=0, encode=Mock(side_effect=[[4, 5], [6]])
        )
    )
    runner.phoneme_frontend = SimpleNamespace(
        metadata={'checkpoint_sha256': 'fixture'},
        generate=Mock(
            side_effect=[
                SimpleNamespace(
                    ipa='wˈʌn', token_ids=(700,), decoder_steps=9, reached_eos=False, stop_reason='max_steps'
                ),
                SimpleNamespace(ipa='tˈuː', token_ids=(701,), decoder_steps=10),
            ]
        ),
    )
    batch = {
        'text': torch.ones(2, 3, dtype=torch.long),
        'raw_texts': ['$1', '2'],
        'languages': ['en', 'en-US'],
        'normalized_text': ['SHOULD NEVER BE READ', 'NOR THIS'],
        'phoneme_tokens': torch.tensor([[999, 999], [998, 998]]),
        'phoneme_tokens_lens': torch.tensor([2, 2]),
    }
    records = runner._set_frontend_phoneme_inputs(batch)
    assert [c.args for c in runner.phoneme_frontend.generate.call_args_list] == [('$1', 'en'), ('2', 'en-US')]
    assert batch['phoneme_tokens'].tolist() == [[100, 4, 5, 101], [100, 6, 101, 0]]
    assert batch['phoneme_tokens_lens'].tolist() == [4, 3]
    assert [r['frontend_phoneme_token_ids'] for r in records] == [[700], [701]]
    assert records[0]['frontend_reached_eos'] is False and records[0]['frontend_stop_reason'] == 'max_steps'
    assert records[1]['frontend_reached_eos'] is True
    assert [r['forced_phoneme_tokens'] for r in records] == [[100, 4, 5, 101], [100, 6, 101]]


@pytest.mark.unit
def test_frontend_new_turn_starts_without_previous_kv_cache(monkeypatch, tmp_path):
    decoder, model = frontend(monkeypatch, tmp_path)
    original_projection = model.phoneme_final_proj

    def alternating_projection(hidden):
        logits = original_projection(hidden)
        logits.fill_(-100.0)
        logits[:, :, 7 if len(model.forwards) % 2 else 81] = 100.0
        return logits

    model.phoneme_final_proj = alternating_projection
    decoder.generate('$1', 'en')
    decoder.generate('$2', 'en')
    assert len(model.forwards) == 4
    for i in (0, 2):
        assert model.forwards[i].get('past_key_values') is None
        assert model.forwards[i + 1]['past_key_values'] == 'cache'
    assert model.forwards[0]['inputs_embeds'].shape == model.forwards[2]['inputs_embeds'].shape


@pytest.mark.unit
def test_disabled_single_turn_frontend_keeps_original_identifier():
    config = EasyMagpieInferenceConfig()
    assert config.phoneme_frontend_nemo_file is None
    assert '_Frontend' not in config.build_identifier()
    assert config.build_identifier().endswith('_Phoneme_gt_argmax')


@pytest.mark.unit
def test_shared_tokenizer_preserves_frontend_ids_and_removes_pad():
    tokenizer = SimpleNamespace(
        tokens={'a': 7, 'b': 8, '<pad>': 0},
        pad=0,
        bos_token_id=100,
        eos_token_id=101,
        unk_token_id=102,
        encode=Mock(side_effect=AssertionError('Shared tokenizer must not encode IPA again')),
    )
    source = SimpleNamespace(**vars(tokenizer))
    source.tokens = dict(tokenizer.tokens)
    frontend = SimpleNamespace(model=SimpleNamespace(phoneme_tokenizer=source))
    output = SimpleNamespace(ipa='ab', token_ids=(0, 7, 0, 8, 0))
    ids, transfer = _frontend_phoneme_tokens(frontend, tokenizer, output)
    assert ids == [100, 7, 8, 101]
    assert transfer == 'direct_ids'
    tokenizer.encode.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize('mismatch', ['vocabulary', 'special_ids'])
def test_different_tokenizer_mapping_uses_ipa_fallback(mismatch):
    source = SimpleNamespace(tokens={'a': 7, '<pad>': 0}, pad=0, bos_token_id=100, eos_token_id=101, unk_token_id=102)
    target = SimpleNamespace(**vars(source), encode=Mock(return_value=[9]))
    if mismatch == 'vocabulary':
        target.tokens = {'a': 9, '<pad>': 0}
    else:
        target.bos_token_id = 200
    ids, transfer = _frontend_phoneme_tokens(
        SimpleNamespace(model=SimpleNamespace(phoneme_tokenizer=source)),
        target,
        SimpleNamespace(ipa='a', token_ids=(7,)),
    )
    assert ids == [target.bos_token_id, 9, 101]
    assert transfer == 'retokenized_ipa'
    target.encode.assert_called_once_with('a')


@pytest.mark.unit
def test_partial_frontend_output_is_flagged_and_cached_without_reference_fallback(monkeypatch, tmp_path):
    decoder, _ = frontend(monkeypatch, tmp_path, emit_eos=False)
    decoder.on_max_steps = 'use_partial'
    result = decoder.generate('$1', 'en')
    assert result.ipa == 'wˈʌn' and result.token_ids == (7, 7)
    assert result.decoder_steps == 2 and not result.reached_eos
    assert result.stop_reason == 'max_steps'
    assert decoder.generate('$1', 'en') is result


@pytest.mark.unit
def test_frontend_rejects_unknown_step_limit_policy():
    with pytest.raises(ValueError, match='on_max_steps'):
        TextOnlyPhonemeFrontend('/unused.nemo', 'cpu', on_max_steps='ignore')


@pytest.mark.unit
def test_cli_can_score_flagged_partial_frontend_predictions():
    args = create_argument_parser().parse_args(
        CLI_PATHS
        + [
            '--model_type',
            'easy_magpie',
            '--phoneme_frontend_nemo_file',
            '/frontend.nemo',
            '--phoneme_frontend_on_max_steps',
            'use_partial',
        ]
    )
    assert _build_easy_magpie_config(args).phoneme_frontend_on_max_steps == 'use_partial'
