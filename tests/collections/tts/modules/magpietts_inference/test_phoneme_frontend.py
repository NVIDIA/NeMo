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
    EasyMagpieMultiturnUserAudioInferenceConfig,
    EasyMagpieMultiturnUserAudioInferenceRunner,
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


@pytest.mark.unit
def test_disabled_frontend_preserves_multiturn_identifier():
    config = EasyMagpieMultiturnUserAudioInferenceConfig()
    assert config.phoneme_frontend_nemo_file is None
    assert config.build_identifier().endswith('_MTUserAudio_MaxTurns6')
    config.phoneme_frontend_nemo_file = '/model/frontend.nemo'
    assert '_FrontendD8_' in config.build_identifier()


@pytest.mark.unit
@pytest.mark.parametrize(
    'options', [[], ['--easy_magpie_inference_mode', 'multiturn_user_audio', '--phoneme_input_type', 'predicted']]
)
def test_cli_rejects_frontend_without_multiturn_forced_phonemes(options):
    args = create_argument_parser().parse_args(
        CLI_PATHS + ['--phoneme_frontend_nemo_file', '/frontend.nemo'] + options
    )
    with pytest.raises(ValueError, match='multiturn_user_audio'):
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
