# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from nemo.collections.tts.modules.magpietts_inference.singleturn_distributed import (
    merge_singleturn_rank_outputs,
    singleturn_rank_indices,
    write_singleturn_rank_manifest,
)


pytestmark = pytest.mark.unit


def _prepare(root, count, world_size, frontend=False):
    for rank in range(world_size):
        folder = root / f'rank_{rank:04d}'
        folder.mkdir()
        indices = singleturn_rank_indices(count, rank, world_size)
        for index in indices:
            (folder / f'predicted_audio_{index}.wav').write_bytes(f'audio-{index}'.encode())
            (folder / f'predicted_codes_{index}.pt').write_bytes(f'codes-{index}'.encode())
            (folder / f'context_audio_{index}.wav').write_bytes(f'context-{index}'.encode())
        if frontend:
            (folder / 'phoneme_frontend_manifest.jsonl').write_text(
                ''.join(
                    json.dumps({'item_idx': i, 'predicted_audio_filepath': 'old', 'phoneme_tokens': [i]}) + '\n'
                    for i in indices
                )
            )
        write_singleturn_rank_manifest(folder, rank, world_size, count, indices, frontend)


@pytest.mark.parametrize('count,world_size', [(13, 8), (2, 8), (17, 3), (0, 2)])
def test_merge_has_exact_original_manifest_order(tmp_path, count, world_size):
    _prepare(tmp_path, count, world_size)
    report = merge_singleturn_rank_outputs(tmp_path, world_size, count, True, timeout_sec=0)
    assert report['source_indices'] == list(range(count))
    assert len(list(tmp_path.glob('predicted_audio_*.wav'))) == count
    for index in range(count):
        assert (tmp_path / f'predicted_audio_{index}.wav').read_bytes() == f'audio-{index}'.encode()
        assert (tmp_path / f'predicted_codes_{index}.pt').read_bytes() == f'codes-{index}'.encode()
        assert (tmp_path / f'context_audio_{index}.wav').read_bytes() == f'context-{index}'.encode()


def test_missing_generated_audio_fails_instead_of_scoring_partial_cohort(tmp_path):
    _prepare(tmp_path, 5, 2)
    (tmp_path / 'rank_0001/predicted_audio_3.wav').unlink()
    with pytest.raises(FileNotFoundError, match='sample 3'):
        merge_singleturn_rank_outputs(tmp_path, 2, 5, False, timeout_sec=0)


def test_stale_completion_markers_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv('TORCHELASTIC_RUN_ID', 'old')
    _prepare(tmp_path, 5, 2)
    monkeypatch.setenv('TORCHELASTIC_RUN_ID', 'new')
    with pytest.raises(RuntimeError, match='Timed out'):
        merge_singleturn_rank_outputs(tmp_path, 2, 5, False, timeout_sec=0)


def test_invalid_shard_cannot_duplicate_or_lose_samples(tmp_path):
    _prepare(tmp_path, 5, 2)
    path = tmp_path / 'rank_0001/singleturn_rank_manifest.json'
    document = json.loads(path.read_text())
    document['source_indices'] = [0, 3]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='shard metadata'):
        merge_singleturn_rank_outputs(tmp_path, 2, 5, False, timeout_sec=0)


def test_frontend_token_ids_and_global_indices_survive_merge(tmp_path):
    _prepare(tmp_path, 5, 2, frontend=True)
    merge_singleturn_rank_outputs(tmp_path, 2, 5, False, timeout_sec=0)
    rows = [json.loads(line) for line in (tmp_path / 'phoneme_frontend_manifest.jsonl').read_text().splitlines()]
    assert [row['item_idx'] for row in rows] == list(range(5))
    for row in rows:
        assert row['phoneme_tokens'] == [row['item_idx']]
        assert Path(row['predicted_audio_filepath']).parent == tmp_path


def test_incomplete_rank_cannot_publish_completion(tmp_path):
    with pytest.raises(ValueError, match='Incomplete'):
        write_singleturn_rank_manifest(tmp_path, 0, 2, 5, [0, 2], False)
