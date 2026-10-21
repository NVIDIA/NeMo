# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rank-local single-turn generation and a validated merge in manifest order."""

import json
import os
import shutil
import time
from pathlib import Path


def singleturn_rank_indices(total_samples, rank, world_size):
    if total_samples < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError('Invalid single-turn shard dimensions')
    return list(range(rank, total_samples, world_size))


def write_singleturn_rank_manifest(output_dir, rank, world_size, total_samples, source_indices, frontend_used):
    expected = singleturn_rank_indices(total_samples, rank, world_size)
    if list(source_indices) != expected:
        raise ValueError(f'Incomplete single-turn rank {rank}: expected {expected}, got {source_indices}')
    path = Path(output_dir) / 'singleturn_rank_manifest.json'
    record = dict(
        run_id=os.environ.get('TORCHELASTIC_RUN_ID', ''),
        rank=rank,
        world_size=world_size,
        total_samples=total_samples,
        source_indices=expected,
        frontend_used=bool(frontend_used),
    )
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record), encoding='utf-8')
    temporary.replace(path)


def merge_singleturn_rank_outputs(repeat_audio_dir, world_size, total_samples, save_predicted_codes, timeout_sec=7200):
    """Wait for this torchrun's ranks, then merge files using original source IDs.

    The original dataset manifest remains the evaluator reference. Missing outputs,
    stale completion markers, and incorrect shard membership cannot be scored as a
    complete evaluation. Empty ranks are valid when there are fewer inputs than GPUs.
    """
    root = Path(repeat_audio_dir)
    run_id = os.environ.get('TORCHELASTIC_RUN_ID', '')
    deadline = time.monotonic() + timeout_sec
    while True:
        documents = []
        for rank in range(world_size):
            path = root / f'rank_{rank:04d}' / 'singleturn_rank_manifest.json'
            try:
                document = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                break
            if document.get('run_id') != run_id:
                break
            documents.append(document)
        if len(documents) == world_size:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Timed out waiting for single-turn rank outputs in {root}')
        time.sleep(min(1, max(0, deadline - time.monotonic())))

    for rank, document in enumerate(documents):
        if (
            document.get('rank') != rank
            or document.get('world_size') != world_size
            or document.get('total_samples') != total_samples
            or document.get('source_indices') != singleturn_rank_indices(total_samples, rank, world_size)
        ):
            raise ValueError(f'Invalid single-turn shard metadata: rank {rank}')
        for index in document['source_indices']:
            source = root / f'rank_{rank:04d}' / f'predicted_audio_{index}.wav'
            if not source.is_file():
                raise FileNotFoundError(f'Missing generated single-turn sample {index}: {source}')
            if save_predicted_codes and not (source.parent / f'predicted_codes_{index}.pt').is_file():
                raise FileNotFoundError(f'Missing generated codec tokens for sample {index}')

    frontend_flags = {document['frontend_used'] for document in documents}
    if len(frontend_flags) != 1:
        raise ValueError('Single-turn ranks disagree on phoneme frontend use')
    frontend_records = []
    for pattern in ['predicted_audio_*.wav', 'predicted_codes_*.pt', 'context_audio_*.wav', 'target_audio_*.wav']:
        for previous in root.glob(pattern):
            previous.unlink()
    for rank, document in enumerate(documents):
        rank_dir = root / f'rank_{rank:04d}'
        for index in document['source_indices']:
            names = [f'predicted_audio_{index}.wav', f'context_audio_{index}.wav', f'target_audio_{index}.wav']
            if save_predicted_codes:
                names.append(f'predicted_codes_{index}.pt')
            for name in names:
                source = rank_dir / name
                if source.is_file():
                    shutil.copyfile(source, root / name)
        if document['frontend_used']:
            with (rank_dir / 'phoneme_frontend_manifest.jsonl').open(encoding='utf-8') as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            if [row['item_idx'] for row in rows] != document['source_indices']:
                raise ValueError(f'Invalid frontend source indices for rank {rank}')
            for row in rows:
                row['predicted_audio_filepath'] = str(root / f'predicted_audio_{row["item_idx"]}.wav')
            frontend_records.extend(rows)
    if frontend_records:
        frontend_records.sort(key=lambda row: row['item_idx'])
        destination = root / 'phoneme_frontend_manifest.jsonl'
        temporary = destination.with_suffix('.jsonl.tmp')
        temporary.write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in frontend_records), encoding='utf-8'
        )
        temporary.replace(destination)
    summary = dict(
        world_size=world_size, total_samples=total_samples, source_indices=list(range(total_samples)), run_id=run_id
    )
    (root / 'singleturn_merged.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
