# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Timing-only API trials: same pairs per worker count, separate from model scores."""
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .evaluate_kimi import KimiEvaluator, summarize
from .runner import _append_json, _atomic_json, _output_lock, load_metric_records


def benchmark_evaluation(specs, output_dir, worker_counts, samples=16, timeout=180, evaluator=None):
    """Benchmark matched requests without reusing or choosing model-score judgments."""
    if not worker_counts or len(set(worker_counts)) != len(worker_counts) or min(worker_counts) < 1 or samples < 1:
        raise ValueError('Benchmark worker counts must be distinct positive integers; samples must be positive')
    evaluator = evaluator or KimiEvaluator(timeout=timeout)
    spec = next(iter(specs.values()))
    path = spec['metrics'][0] if isinstance(spec['metrics'], list) else spec['metrics']
    rows = [
        row
        for row in load_metric_records(
            path, spec.get('reference_manifest'), spec.get('source_field'), spec.get('asr_field', 'pred_text')
        )
        if row['input_error'] is None
    ]
    rows = random.Random(42).sample(rows, min(samples, len(rows)))
    if len(rows) < max(worker_counts):
        raise ValueError('Need at least as many benchmark samples as the largest worker count')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with _output_lock(output_dir):
        if (output_dir / 'benchmark_judgments.jsonl').exists():
            raise ValueError('Benchmark output already exists; use a new directory for a new timing trial')
        _atomic_json(output_dir / 'benchmark_inputs.json', rows)
        trials = []
        for workers in worker_counts:
            started = time.monotonic()
            results = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(evaluator.evaluate_safe, row['source_text'], row['asr_transcript']): row
                    for row in rows
                }
                for future in as_completed(futures):
                    row = futures[future]
                    result = future.result()
                    results.append(result)
                    _append_json(
                        output_dir / 'benchmark_judgments.jsonl',
                        dict(result, id=row['id'], workers=workers, timing_only=True),
                    )
            elapsed = time.monotonic() - started
            summary = summarize(results)
            errors = [result['error'] for result in results if result['error']]
            trial = dict(
                workers=workers,
                requests=len(rows),
                elapsed_seconds=elapsed,
                requests_per_second=len(rows) / elapsed,
                request_or_validation_errors=len(errors),
                rate_limited_requests=sum(error.get('http_status') == 429 for error in errors),
                scored=summary['scored'],
                reviews=summary['judge_review_count'],
            )
            serial = next((item['elapsed_seconds'] for item in trials if item['workers'] == 1), None)
            trial['speedup_vs_serial'] = serial / elapsed if serial is not None else 1.0 if workers == 1 else None
            trials.append(trial)
            _atomic_json(output_dir / 'benchmark_progress.json', dict(trials=trials))
            print('API_BENCHMARK', json.dumps(trial), flush=True)
        clean = [trial for trial in trials if trial['request_or_validation_errors'] == 0]
        recommended = max(clean, key=lambda trial: trial['requests_per_second'])['workers'] if clean else None
        report = dict(
            complete=True,
            samples_per_trial=len(rows),
            seed=42,
            prompt_sha256=evaluator.prompt_sha256,
            trials=trials,
            recommended_workers=recommended,
            timing_only=True,
            repeated_judgments_excluded_from_model_scores=True,
        )
        _atomic_json(output_dir / 'benchmark_summary.json', report)
        return report
