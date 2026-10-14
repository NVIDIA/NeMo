# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from nemo.collections.tts.modules.magpietts_inference.evaluate_generated_audio import load_evalset_config

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('key', ['speaker_a', '__zero__'])
def test_evalset_accepts_precomputed_context_key(tmp_path, key):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('', encoding='utf-8')
    info = {'manifest_path': 'manifest.json', 'audio_dir': '.', 'precomputed_context_audio_embedding_key': key}
    config = tmp_path / 'evalset.json'
    config.write_text(json.dumps({'test': info}), encoding='utf-8')

    loaded = load_evalset_config(str(config), dataset_base_path=tmp_path)

    assert loaded['test']['precomputed_context_audio_embedding_key'] == key
    assert loaded['test']['manifest_path'] == str(manifest)


@pytest.mark.parametrize('key', [None, '', ' ', ' speaker_a', 123])
def test_evalset_rejects_invalid_precomputed_context_key(tmp_path, key):
    config = tmp_path / 'evalset.json'
    config.write_text(
        json.dumps(
            {
                'test': {
                    'manifest_path': 'manifest.json',
                    'audio_dir': '.',
                    'precomputed_context_audio_embedding_key': key,
                }
            }
        ),
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='precomputed_context_audio_embedding_key'):
        load_evalset_config(str(config), dataset_base_path=tmp_path)
