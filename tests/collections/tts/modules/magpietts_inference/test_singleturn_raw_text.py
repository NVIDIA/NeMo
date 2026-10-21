# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Raw single-turn TN inputs must remain separate from normalized evaluation references."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from nemo.collections.tts.modules.magpietts_inference.evaluate_generated_audio import load_evalset_config
from nemo.collections.tts.modules.magpietts_inference.inference import EasyMagpieInferenceRunner


@pytest.mark.unit
@pytest.mark.parametrize(
    'probability,expected', [(None, 'Pay one dollar.'), (0.0, 'Pay $1.'), (1.0, 'Pay one dollar.')]
)
def test_singleturn_raw_input_keeps_normalized_reference(tmp_path, probability, expected):
    manifest = tmp_path / 'eval.json'
    record = {
        'audio_filepath': 'target.wav',
        'context_audio_filepath': 'reference.wav',
        'duration': 1.0,
        'context_audio_duration': 3.0,
        'text': 'Pay $1.',
        'original_text': 'Pay $1.',
        'normalized_text': 'Pay one dollar.',
        'ipa': 'pˈeɪ wˈʌn dˈɑːlɚ.',
    }
    manifest.write_text(json.dumps(record) + '\n', encoding='utf-8')
    meta = {'manifest_path': str(manifest), 'audio_dir': str(tmp_path), 'language': 'en'}
    if probability is not None:
        meta['load_normalized_text_percent'] = probability
    runner = object.__new__(EasyMagpieInferenceRunner)
    runner._get_context_durations = Mock(return_value=(3.0, 10.0))
    runner.config = SimpleNamespace(default_tokenizer_name='nemotron_nano_30b')
    runner.model = SimpleNamespace(
        sample_rate=22050,
        codec_model_samples_per_frame=256,
        eos_id=100,
        num_audio_codebooks=8,
        text_conditioning_tokenizer_name='nemotron_nano_30b',
        enable_phoneme_text_input=False,
        text_phoneme_token_offset=None,
        phoneme_text_bop_marker='<bop>',
        phoneme_text_eop_marker='<eop>',
        add_language_to_context_text=True,
        tokenizer=object(),
        cfg=OmegaConf.create({}),
    )
    dataset = runner.create_dataset({'tn': meta})
    assert len(dataset) == 1
    assert dataset.data_samples[0].text == expected
    assert dataset.data_samples[0].manifest_entry['text'] == 'Pay $1.'
    assert dataset.data_samples[0].manifest_entry['normalized_text'] == 'Pay one dollar.'
    assert runner._manifest_records[0]['normalized_text'] == 'Pay one dollar.'
    assert meta.get('load_normalized_text_percent') == probability


@pytest.mark.unit
@pytest.mark.parametrize('probability', [0, 0.0, 0.5, 1, 1.0])
def test_singleturn_config_accepts_input_normalization_probability(tmp_path, probability):
    (tmp_path / 'eval.json').write_text('{}\n', encoding='utf-8')
    entry = {
        'manifest_path': str(tmp_path / 'eval.json'),
        'audio_dir': str(tmp_path),
        'load_normalized_text_percent': probability,
    }
    path = tmp_path / 'datasets.json'
    path.write_text(json.dumps({'tn': entry}), encoding='utf-8')
    loaded = load_evalset_config(str(path), dataset_base_path=tmp_path)
    assert loaded['tn']['load_normalized_text_percent'] == probability


@pytest.mark.unit
@pytest.mark.parametrize('probability', [None, True, False, -0.1, 1.1, '0.0', float('nan')])
def test_singleturn_config_rejects_invalid_input_normalization_probability(tmp_path, probability):
    (tmp_path / 'eval.json').write_text('{}\n', encoding='utf-8')
    entry = {
        'manifest_path': str(tmp_path / 'eval.json'),
        'audio_dir': str(tmp_path),
        'load_normalized_text_percent': probability,
    }
    path = tmp_path / 'datasets.json'
    path.write_text(json.dumps({'tn': entry}), encoding='utf-8')
    with pytest.raises(ValueError, match='load_normalized_text_percent'):
        load_evalset_config(str(path), dataset_base_path=tmp_path)
