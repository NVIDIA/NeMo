# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from examples.tts.magpietts_inference import run_inference_and_evaluation

from nemo.collections.tts.modules.magpietts_inference.evaluate_generated_audio import _get_record_texts
from nemo.collections.tts.modules.magpietts_inference.evaluation import EvaluationConfig
from nemo.collections.tts.modules.magpietts_inference.inference import (
    EasyMagpieInferenceConfig,
    EasyMagpieInferenceRunner,
)
from nemo.collections.tts.modules.magpietts_inference.singleturn_distributed import (
    merge_singleturn_rank_outputs,
    write_singleturn_rank_manifest,
)

pytestmark = pytest.mark.unit
MANIFEST = 'singleturn_evaluation_manifest.jsonl'


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, selected, records):
        self.data_samples = [SimpleNamespace(text=text) for text in selected]
        self.records = records

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        return index

    def collate_fn(self, indices):
        tokens = [torch.tensor(list(map(ord, self.data_samples[i].text))) for i in indices]
        return dict(
            text=torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True),
            text_lens=torch.tensor([len(t) for t in tokens]),
            raw_texts=[self.records[i]['text'] for i in indices],
            languages=['en'] * len(indices),
        )


def _runner(frontend=False):
    runner = object.__new__(EasyMagpieInferenceRunner)
    runner.config = EasyMagpieInferenceConfig(batch_size=2)
    runner._batch_to_cuda = lambda batch: batch
    runner._copy_reference_audio = Mock()
    runner._current_precomputed_audio_embedding = None
    runner.phoneme_frontend = object() if frontend else None
    runner._set_frontend_phoneme_inputs = lambda batch: [dict(text=text) for text in batch['raw_texts']]
    runner.seen_texts = []

    def infer(batch, **kwargs):
        runner.seen_texts.extend(
            ''.join(map(chr, ids[:length].tolist())) for ids, length in zip(batch['text'], batch['text_lens'])
        )
        size = batch['text'].size(0)
        return SimpleNamespace(
            predicted_audio=torch.zeros(size, 8),
            predicted_audio_lens=torch.full((size,), 8),
            predicted_codes=torch.zeros(size, 1, 1),
            predicted_codes_lens=torch.ones(size, dtype=torch.long),
            rtf_metrics={},
        )

    runner.model = SimpleNamespace(sample_rate=22050, infer_batch=infer)
    return runner


def _records():
    return [
        dict(
            id=f'sample-{i}',
            audio_filepath=f'target_{i}.wav',
            text=f'Pay ${i+1}.',
            original_text=f'Pay ${i+1}.',
            normalized_text=f'Pay {word} dollars.',
        )
        for i, word in enumerate(['one', 'two', 'three'])
    ]


@pytest.mark.parametrize('normalized', [False, True])
@pytest.mark.parametrize('frontend', [False, True])
def test_singleturn_logs_selected_text(tmp_path, normalized, frontend):
    records = _records()
    selected = [row['normalized_text' if normalized else 'text'] for row in records]
    runner = _runner(frontend)
    runner._run_decoder_only_inference(
        _Dataset(selected, records), str(tmp_path), records, str(tmp_path), False, False
    )
    path = tmp_path / MANIFEST
    assert path.is_file(), 'Missing manifest of the actual single-turn text input'
    logged = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row['text'] for row in logged] == runner.seen_texts == selected
    assert [row['original_text'] for row in logged] == [row['text'] for row in records]
    assert [row['normalized_text'] for row in logged] == [row['normalized_text'] for row in records]
    assert [row['item_idx'] for row in logged] == [0, 1, 2]
    assert runner.evaluation_manifest_path == str(path)
    assert runner.evaluation_audio_dir == str(tmp_path)
    assert records == _records(), 'Logging must not mutate the original manifest'
    for row, expected in zip(logged, selected):
        tts_text, normalized_reference, metric_reference = _get_record_texts(row)
        assert tts_text == expected
        assert normalized_reference == metric_reference == row['normalized_text']


@pytest.mark.parametrize('distributed', [False, True])
def test_entrypoint_scores_the_actual_input_manifest(tmp_path, monkeypatch, distributed):
    records = _records()
    selected = [row['normalized_text'] for row in records]
    dataset = _Dataset(selected, records)
    runner = _runner()
    source = tmp_path / 'source.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in records))
    runner.create_dataset = Mock(return_value=dataset)
    original_run = runner.run_inference_on_dataset
    module = 'examples.tts.magpietts_inference'
    monkeypatch.setattr(f'{module}._get_torchrun_rank_info', lambda: (0, 2 if distributed else 1, 0))
    if distributed:

        def run_both(**kwargs):
            other = _runner()
            other.set_distributed_context(1, 2)
            other.run_inference_on_dataset(
                **dict(kwargs, output_dir=str(Path(kwargs['output_dir']).parent / 'rank_0001'))
            )
            return original_run(**kwargs)

        runner.run_inference_on_dataset = run_both
    seen = []

    def evaluate(manifest_path, audio_dir, generated_audio_dir, config):
        path = Path(manifest_path)
        assert path == Path(generated_audio_dir) / MANIFEST
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row['text'] for row in rows] == selected
        assert [row['original_text'] for row in rows] == [row['text'] for row in records]
        assert [row['normalized_text'] for row in rows] == [row['normalized_text'] for row in records]
        seen.append(rows)
        return dict(cer_cumulative=0.0, ssim_pred_context_avg=1.0), [dict(cer=0.0)]

    monkeypatch.setattr(f'{module}.evaluate_generated_audio_dir', evaluate)
    for name in [
        'create_violin_plot',
        'create_combined_box_plot',
        'append_metrics_to_csv',
        'write_csv_header_if_needed',
    ]:
        monkeypatch.setattr(f'{module}.{name}', lambda *args, **kwargs: None)
    run_inference_and_evaluation(
        runner=runner,
        checkpoint_name='test',
        inference_config=runner.config,
        eval_config=EvaluationConfig(with_fcd=False, with_utmosv2=False),
        dataset_meta_info={'sample': dict(manifest_path=str(source), audio_dir=str(tmp_path), language='en')},
        datasets=['sample'],
        out_dir=str(tmp_path / 'outputs'),
        flops_per_component={},
        moe_info='',
        num_repeats=2,
    )
    assert len(seen) == 2


@pytest.mark.parametrize('count,world_size', [(3, 8), (5, 2)])
def test_input_manifest_merge_preserves_text_and_references(tmp_path, count, world_size):
    for rank in range(world_size):
        folder = tmp_path / f'rank_{rank:04d}'
        folder.mkdir()
        indices = list(range(rank, count, world_size))
        rows = []
        for i in indices:
            (folder / f'predicted_audio_{i}.wav').write_bytes(b'audio')
            rows.append(
                dict(
                    item_idx=i,
                    text=f'number {i}',
                    original_text=str(i),
                    normalized_text=f'number {i}',
                    predicted_audio_filepath=str(folder / f'predicted_audio_{i}.wav'),
                )
            )
        (folder / MANIFEST).write_text(''.join(json.dumps(row) + '\n' for row in rows))
        write_singleturn_rank_manifest(folder, rank, world_size, count, indices, False)
    report = merge_singleturn_rank_outputs(tmp_path, world_size, count, False, timeout_sec=0)
    assert report['evaluation_manifest_path'] == str(tmp_path / MANIFEST)
    rows = [json.loads(line) for line in (tmp_path / MANIFEST).read_text().splitlines()]
    assert [row['item_idx'] for row in rows] == list(range(count))
    assert [row['text'] for row in rows] == [f'number {i}' for i in range(count)]
    assert [row['original_text'] for row in rows] == list(map(str, range(count)))
    assert all(Path(row['predicted_audio_filepath']).parent == tmp_path for row in rows)
