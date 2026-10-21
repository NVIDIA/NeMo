# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel
from nemo.collections.tts.modules.magpietts_inference.inference import (
    BaseInferenceRunner,
    EasyMagpieInferenceConfig,
    EasyMagpieInferenceRunner,
)
from nemo.collections.tts.modules.magpietts_inference.utils import (
    ModelLoadConfig,
    _add_easy_magpie_args,
    load_easy_magpie_model,
)


pytestmark = pytest.mark.unit


class _Codec:
    def __init__(self):
        self.calls = []

    def codes_to_audio(self, codes, lens):
        assert codes.size(-1) > 0
        self.calls.append((codes.size(0), codes.size(-1)))
        audio = codes.float().sum(1).repeat_interleave(2, dim=1)
        lengths = lens * 2
        audio *= torch.arange(audio.size(1))[None, :] < lengths[:, None]
        return audio, lengths, codes


def _finalize(decode_batch_size, lengths):
    batch_size = len(lengths)
    lens = torch.tensor(lengths)
    codes = torch.arange(batch_size * 2 * max(lengths)).reshape(batch_size, 2, max(lengths)) % 23
    codec = _Codec()
    model = SimpleNamespace(
        phoneme_tokenizer=None,
        num_audio_codebooks=2,
        codec_decode_batch_size=decode_batch_size,
        _codec_helper=codec,
        _prepare_codes_for_decode=lambda codes, lens: (codes, lens),
    )
    state = SimpleNamespace(
        config=SimpleNamespace(batch_size=batch_size, device='cpu'),
        all_predictions=[codes],
        all_phoneme_predictions=[],
        audio_prediction_start_idx=torch.zeros(batch_size, dtype=torch.long),
        audio_prediction_end_idx=lens,
    )
    return EasyMagpieTTSInferenceModel.streaming_finalize(model, state), codec.calls


@pytest.mark.parametrize('decode_batch_size', [1, 2, 3])
def test_codec_sub_batches_preserve_valid_audio_and_codes(decode_batch_size):
    lengths = [11, 0, 4, 2, 7]
    reference, _ = _finalize(0, lengths)
    output, calls = _finalize(decode_batch_size, lengths)
    torch.testing.assert_close(output.audio_len, reference.audio_len)
    torch.testing.assert_close(output.audio_codes_len, reference.audio_codes_len)
    torch.testing.assert_close(output.audio_codes, reference.audio_codes)
    for item, length in enumerate(output.audio_len):
        torch.testing.assert_close(output.audio[item, :length], reference.audio[item, :length])
    assert max(batch for batch, _ in calls) <= decode_batch_size
    if decode_batch_size == 1:
        assert calls == [(1, 11), (1, 4), (1, 2), (1, 7)]


@pytest.mark.parametrize('decode_batch_size', [0, 5, 20])
def test_full_batch_decode_path_is_unchanged(decode_batch_size):
    _, calls = _finalize(decode_batch_size, [11, 0, 4, 2, 7])
    assert calls == [(5, 11)]


@pytest.mark.parametrize('override,expected', [(None, 4), (0, 0), (1, 1)])
def test_runner_preserves_checkpoint_setting_unless_overridden(override, expected):
    model = SimpleNamespace(codec_decode_batch_size=4)
    config = EasyMagpieInferenceConfig(codec_decode_batch_size=override)
    with (
        patch.object(BaseInferenceRunner, '_configure_tokenizer'),
        patch('nemo.collections.tts.modules.magpietts_inference.inference._build_phoneme_frontend', return_value=None),
    ):
        EasyMagpieInferenceRunner(model, config)
    assert model.codec_decode_batch_size == expected


def test_negative_decode_batch_size_is_rejected():
    with patch.object(BaseInferenceRunner, '_configure_tokenizer'), pytest.raises(ValueError):
        EasyMagpieInferenceRunner(SimpleNamespace(), EasyMagpieInferenceConfig(codec_decode_batch_size=-1))


def test_cli_decode_override_is_optional():
    import argparse

    parser = argparse.ArgumentParser()
    _add_easy_magpie_args(parser)
    assert parser.parse_args([]).codec_decode_batch_size is None
    assert parser.parse_args(['--codec_decode_batch_size', '1']).codec_decode_batch_size == 1


@pytest.mark.parametrize('device,expected', [('cuda', 'cuda:3'), ('cuda:2', 'cuda:2'), ('cpu', 'cpu')])
def test_checkpoint_restore_uses_assigned_device(device, expected):
    model = MagicMock()
    config = OmegaConf.create({})
    with (
        patch('torch.cuda.current_device', return_value=3),
        patch('nemo.collections.tts.modules.magpietts_inference.utils.EasyMagpieTTSInferenceModel') as model_class,
    ):
        model_class.restore_from.side_effect = [config, model]
        restored, _ = load_easy_magpie_model(ModelLoadConfig(nemo_file='test.nemo'), device=device)
    assert restored is model
    for call in model_class.restore_from.call_args_list:
        assert call.kwargs['map_location'] == torch.device(expected)
    model.to.assert_called_once_with(torch.device(expected))


@pytest.mark.parametrize('mode', ['single_turn', 'multiturn_user_audio'])
def test_entrypoint_selects_cuda_for_both_easy_modes(mode):
    import ast

    source = Path(__file__).resolve().parents[5] / 'examples/tts/magpietts_inference.py'
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    # Run argument parsing and device selection, stopping before model loading.
    function.body = function.body[:5]
    configure = MagicMock()
    args = SimpleNamespace(model_type='easy_magpie', easy_magpie_inference_mode=mode, batch_size=1)
    namespace = dict(
        create_argument_parser=lambda: SimpleNamespace(parse_args=lambda argv: args),
        _configure_cuda_for_rank=configure,
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    namespace['main']([])
    configure.assert_called_once_with()
