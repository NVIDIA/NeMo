# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU/API scoring of saved NeMo or Qwen TTS evaluation JSON; no audio inference."""
import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .evaluate_kimi import API_URL, MODEL, KimiEvaluator, summarize
from .validate_judgment import reject_duplicate_keys


def load_metric_records(path, reference_manifest=None, source_field=None, asr_field='pred_text'):
    """Keep every metric row; resolve raw source by stable ID or reference-audio path.

    Never use gt_text, normalized_text, dataloader_normalized_text or gt_audio_text
    as the source. Empty ASR is a valid observation; absent/non-string ASR is not.
    """
    path = Path(path).resolve()
    refs_by_id, refs_by_audio = {}, {}
    if reference_manifest is not None:
        manifest = Path(reference_manifest).resolve()
        for source in _read_rows(manifest):
            for key in ('id', 'cut_id'):
                if isinstance(source.get(key), str):
                    _index_unique(refs_by_id, source[key], source)
            audio = source.get('audio_filepath')
            if isinstance(audio, str):
                _index_unique(refs_by_audio, _audio_key(audio, manifest.parent), source)
    records = []
    for index, row in enumerate(_read_rows(path)):
        source, origin = None, None
        identifier = row.get('id', row.get('cut_id'))
        reference = refs_by_id.get(identifier) if isinstance(identifier, str) else None
        audio = row.get('gt_audio_filepath')
        if isinstance(audio, str):
            audio_reference = refs_by_audio.get(_audio_key(audio, path.parent))
            if reference is not None and audio_reference is not None and reference != audio_reference:
                raise ValueError('Conflicting ID and audio reference at metric row ' + str(index))
            reference = reference if reference is not None else audio_reference
        if source_field:
            source, origin = row.get(source_field), source_field
        elif reference is not None:
            source, origin = reference.get('original_text', reference.get('text')), 'reference_manifest'
            identifier = reference.get('id', reference.get('cut_id', identifier))
        else:
            for field in ('source_text', 'raw_text', 'original_text', 'tts_text_input'):
                if isinstance(row.get(field), str):
                    value = row[field]
                    ipa = row.get('phoneme_frontend_as_text_input', False) or '<bop>' in value or '[IPA]' in value
                    if field == 'tts_text_input' and (ipa or '_IPATextInput' in str(path)):
                        continue
                    source, origin = value, field
                    break
        error = None
        transcript = row.get(asr_field)
        if not isinstance(source, str) or not source.strip():
            error = {'type': 'MissingRawSourceText', 'http_status': None}
        elif not isinstance(transcript, str):
            error = {'type': 'MissingASRTranscript', 'http_status': None}
        records.append(
            dict(
                id=identifier if isinstance(identifier, str) else str(index),
                metrics_row=index,
                metrics_path=str(path),
                source_text=source,
                asr_transcript=transcript,
                source_origin=origin,
                input_error=error,
            )
        )
    return records


def run_evaluation(
    specs,
    output_dir,
    workers=4,
    max_attempts=4,
    timeout=180,
    evaluator=None,
    retry_invalid=False,
    reuse_results=None,
):
    """Score all rows, durably reuse exact-pair judgments, and summarize each model.

    Transport failures receive bounded retries. Opt in to retrying malformed or
    invalid judgments; the first valid score or REVIEW is always final. A compatible
    completed run can seed a new output directory without rejudging valid results.
    """
    if workers < 1 or max_attempts < 1 or timeout <= 0:
        raise ValueError('workers, max_attempts and timeout must be positive')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with _output_lock(output_dir):
        evaluator = evaluator or KimiEvaluator(timeout=timeout)
        inputs, file_hashes = [], {}
        for name, spec in specs.items():
            if not isinstance(name, str) or not name:
                raise ValueError('Each model needs a nonempty name')
            paths = spec['metrics'] if isinstance(spec['metrics'], list) else [spec['metrics']]
            if not paths or len(set(map(str, paths))) != len(paths):
                raise ValueError('Metrics paths must be nonempty and unique within each model')
            manifest = spec.get('reference_manifest')
            if manifest:
                file_hashes[str(Path(manifest).resolve())] = _file_sha256(manifest)
            for repeat, path in enumerate(paths):
                file_hashes[str(Path(path).resolve())] = _file_sha256(path)
                rows = load_metric_records(
                    path, manifest, spec.get('source_field'), spec.get('asr_field', 'pred_text')
                )
                for row in rows:
                    row.update(
                        model=name,
                        label=spec.get('label', name),
                        repeat=repeat,
                        record_key=_hash_json([name, repeat, row['metrics_row']]),
                    )
                inputs.extend(rows)
        if not inputs:
            raise ValueError('No evaluation records')
        identity = dict(
            version=1,
            inputs=specs,
            file_sha256=file_hashes,
            prompt_sha256=evaluator.prompt_sha256,
            endpoint=API_URL,
            judge_model=MODEL,
            temperature=0,
            top_p=1,
            reasoning_effort='low',
            max_tokens=8192,
            stream=False,
            retry_invalid=retry_invalid,
            max_attempts=max_attempts,
            implementation_sha256={
                p.name: _file_sha256(p)
                for p in [
                    Path(__file__),
                    Path(__file__).with_name('evaluate_kimi.py'),
                    Path(__file__).with_name('validate_judgment.py'),
                ]
            },
        )
        if reuse_results is not None:
            reuse_results = Path(reuse_results).resolve()
            if reuse_results == output_dir.resolve():
                raise ValueError('Reuse requires a separate new output directory')
            previous = _validate_reuse(reuse_results, identity)
            identity['reuse_results'] = dict(path=str(reuse_results), fingerprint=previous['fingerprint'])
        fingerprint = _hash_json(identity)
        manifest_path = output_dir / 'run.json'
        if manifest_path.exists():
            if _strict_load(manifest_path)['fingerprint'] != fingerprint:
                raise ValueError('Inputs, prompt or implementation changed; use a new output directory')
        else:
            _atomic_json(
                manifest_path,
                dict(
                    identity,
                    fingerprint=fingerprint,
                    created_at=_now(),
                    records=len(inputs),
                    workers=workers,
                    max_attempts=max_attempts,
                    timeout_seconds=timeout,
                ),
            )
        _atomic_json(output_dir / 'input_records.json', inputs)
        cache_dir = output_dir / 'pair_cache'
        cache_dir.mkdir(exist_ok=True)
        if reuse_results is not None:
            _seed_pair_cache(reuse_results, cache_dir, inputs, evaluator.prompt_sha256, identity['reuse_results'])
        journal = output_dir / 'judgments.jsonl'
        finished = _read_journal(journal)
        by_key = {row['record_key']: row for row in inputs}
        if not set(finished) <= set(by_key):
            raise ValueError('Output journal contains unknown records')
        for key, saved in finished.items():
            _validate_saved(saved, by_key[key], evaluator.prompt_sha256)
        groups = defaultdict(list)
        for row in inputs:
            if row['record_key'] in finished:
                continue
            if row['input_error']:
                result = dict(
                    source_text=row['source_text'],
                    asr_transcript=row['asr_transcript'],
                    score_1_to_5=None,
                    judgment=None,
                    error=row['input_error'],
                    audit=dict(requested_model=MODEL, prompt_sha256=evaluator.prompt_sha256),
                )
                saved = dict(result, **_record_metadata(row), cache_hit=False)
                _append_json(journal, saved)
                finished[row['record_key']] = saved
            else:
                pair_key = _hash_json(
                    [evaluator.prompt_sha256, API_URL, MODEL, row['source_text'], row['asr_transcript']]
                )
                groups[pair_key].append(row)
        attempts_path = output_dir / 'attempts.jsonl'
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_judge_pair, key, rows[0], evaluator, cache_dir, max_attempts, retry_invalid): key
                for key, rows in groups.items()
            }
            for future in as_completed(futures):
                pair_key = futures[future]
                result, attempts, cache_hit = future.result()
                for attempt in attempts:
                    _append_json(attempts_path, attempt)
                for row in groups[pair_key]:
                    saved = dict(result, **_record_metadata(row), cache_hit=cache_hit, pair_key=pair_key)
                    _append_json(journal, saved)
                    finished[row['record_key']] = saved
                if len(finished) % 100 < len(groups[pair_key]) or len(finished) == len(inputs):
                    print(json.dumps(dict(event='progress', completed=len(finished), total=len(inputs))), flush=True)
        if set(finished) != set(by_key):
            raise ValueError('Final output coverage does not match the declared input rows')
        model_results = {}
        for name, spec in specs.items():
            rows = [finished[row['record_key']] for row in inputs if row['model'] == name]
            summary = summarize(rows)
            summary.update(
                label=spec.get('label', name),
                by_repeat={
                    str(r): summarize([row for row in rows if row['repeat'] == r])
                    for r in sorted({row['repeat'] for row in rows})
                },
                input_error_count=sum(
                    row['source_origin'] is None
                    or row['error'] is not None
                    and row['error']['type'] in {'MissingRawSourceText', 'MissingASRTranscript'}
                    for row in rows
                ),
            )
            summary['entity_checks'] = dict(
                Counter(
                    row['judgment']['checks']['values_and_entities'] for row in rows if row['judgment'] is not None
                )
            )
            model_results[name] = summary
        cohorts = {
            name: _hash_json(
                sorted(
                    row['source_text']
                    for row in inputs
                    if row['model'] == name and row['repeat'] == 0 and isinstance(row['source_text'], str)
                )
            )
            for name in specs
        }
        output = dict(
            complete=True,
            records=len(inputs),
            prompt_sha256=evaluator.prompt_sha256,
            judge_model=MODEL,
            fingerprint=fingerprint,
            finished_at=_now(),
            same_source_cohort_first_repeat=len(set(cohorts.values())) == 1,
            models=model_results,
        )
        _atomic_json(output_dir / 'summary.json', output)
        _write_table(output_dir / 'summary.txt', model_results)
        print((output_dir / 'summary.txt').read_text(encoding='utf-8'), flush=True)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--metrics', type=Path, nargs='+', help='Saved filewise_metrics JSON or JSONL for one model')
    group.add_argument(
        '--inputs', type=Path, help='JSON object: model name -> metrics path(s), label and optional fields'
    )
    parser.add_argument('--model-name', default='tts_model')
    parser.add_argument('--reference-manifest', type=Path, help='Raw source JSONL, joined by ID or gt_audio_filepath')
    parser.add_argument('--source-field', help='Explicit raw-source field; never point this to normalized text or IPA')
    parser.add_argument('--asr-field', default='pred_text')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-attempts', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument(
        '--retry-invalid',
        action='store_true',
        help='Retry malformed/invalid judgments up to max-attempts; never retry valid scores or REVIEW',
    )
    parser.add_argument(
        '--reuse-results',
        type=Path,
        help='Seed a new output directory from a compatible completed run; preserve every API attempt',
    )
    parser.add_argument('--benchmark-workers', type=int, nargs='+', help='Timing-only worker counts, e.g. 1 4 8 16')
    parser.add_argument('--benchmark-samples', type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1 or args.max_attempts < 1 or args.timeout <= 0:
        parser.error('workers, max-attempts and timeout must be positive')
    if args.inputs:
        specs = _strict_load(args.inputs)
        if not isinstance(specs, dict) or not specs:
            parser.error('--inputs must contain a nonempty model specification object')
    else:
        specs = {args.model_name: dict(metrics=[str(p.resolve()) for p in args.metrics])}
    for spec in specs.values():
        if args.reference_manifest:
            spec.setdefault('reference_manifest', str(args.reference_manifest.resolve()))
        if args.source_field:
            spec.setdefault('source_field', args.source_field)
        spec.setdefault('asr_field', args.asr_field)
    if args.benchmark_workers:
        if args.retry_invalid or args.reuse_results is not None:
            parser.error('Timing-only benchmarks cannot retry or reuse model-score judgments')
        from .benchmark import benchmark_evaluation

        benchmark_evaluation(specs, args.output_dir, args.benchmark_workers, args.benchmark_samples, args.timeout)
        return
    run_evaluation(
        specs,
        args.output_dir,
        args.workers,
        args.max_attempts,
        args.timeout,
        retry_invalid=args.retry_invalid,
        reuse_results=args.reuse_results,
    )


def _judge_pair(key, row, evaluator, cache_dir, max_attempts, retry_invalid=False):
    path = cache_dir / (key + '.json')
    cached = None
    if path.exists():
        cached = _strict_load(path)
        if cached['pair_key'] != key:
            raise ValueError('Cache key mismatch')
        _validate_saved(cached['result'], row, evaluator.prompt_sha256)
        if not (retry_invalid and _invalid_response(cached['result']['error'])):
            return cached['result'], cached.get('attempts', []), True
    attempts = []
    previous = []
    for number in range(1, max_attempts + 1):
        attempt_path = cache_dir / (key + '.attempt_' + str(number) + '.json')
        if not attempt_path.exists():
            break
        saved = _strict_load(attempt_path)
        _validate_saved(saved['result'], row, evaluator.prompt_sha256)
        if saved['attempt']['pair_key'] != key or saved['attempt']['attempt'] != number:
            raise ValueError('Attempt history identity mismatch')
        previous.append(saved)
        attempts.append(saved['attempt'])
        if not _retryable(saved['result']['error'], retry_invalid):
            break
    if cached is not None and not previous:
        raise ValueError('Cannot retry a cached invalid response without its original attempt history')
    if previous and (not _retryable(previous[-1]['result']['error'], retry_invalid) or len(previous) == max_attempts):
        result = previous[-1]['result']
        _atomic_json(path, dict(pair_key=key, result=result, attempts=attempts))
        return result, attempts, True
    for attempt in range(len(previous) + 1, max_attempts + 1):
        started = time.monotonic()
        result = evaluator.evaluate_safe(row['source_text'], row['asr_transcript'])
        attempts.append(
            dict(
                pair_key=key,
                attempt=attempt,
                at=_now(),
                error=result['error'],
                elapsed_seconds=time.monotonic() - started,
            )
        )
        _atomic_json(
            cache_dir / (key + '.attempt_' + str(attempt) + '.json'), dict(attempt=attempts[-1], result=result)
        )
        if not _retryable(result.get('error'), retry_invalid) or attempt == max_attempts:
            break
        if _transient(result['error']):
            time.sleep(min(60, 2 ** (attempt - 1)) + random.random())
        else:
            time.sleep(random.random())
    _validate_saved(result, row, evaluator.prompt_sha256)
    _atomic_json(path, dict(pair_key=key, result=result, attempts=attempts))
    return result, attempts, False


def _invalid_response(error):
    return error is not None and error.get('type') in {'InvalidJudgmentError', 'JSONDecodeError'}


def _retryable(error, retry_invalid):
    return _transient(error) or retry_invalid and _invalid_response(error)


def _validate_reuse(source_dir, identity):
    previous = _strict_load(source_dir / 'run.json')
    summary = _strict_load(source_dir / 'summary.json')
    if not summary.get('complete') or summary['fingerprint'] != previous['fingerprint']:
        raise ValueError('Reuse requires an internally consistent completed source run')
    for key in (
        'version',
        'inputs',
        'file_sha256',
        'prompt_sha256',
        'endpoint',
        'judge_model',
        'temperature',
        'top_p',
        'reasoning_effort',
        'max_tokens',
        'stream',
    ):
        if previous.get(key) != identity[key]:
            raise ValueError('Incompatible reused results: ' + key)
    # Only the runner/retry policy may change; the API client and validator must match.
    for name in ('evaluate_kimi.py', 'validate_judgment.py'):
        if previous['implementation_sha256'].get(name) != identity['implementation_sha256'][name]:
            raise ValueError('Incompatible reused implementation: ' + name)
    return previous


def _seed_pair_cache(source_dir, cache_dir, inputs, prompt_sha256, provenance):
    marker = cache_dir.parent / 'reused_results.json'
    if marker.exists():
        if _strict_load(marker)['source'] != provenance:
            raise ValueError('Reused cache provenance mismatch')
        return
    expected = {}
    for row in inputs:
        if row['input_error'] is None:
            key = _hash_json([prompt_sha256, API_URL, MODEL, row['source_text'], row['asr_transcript']])
            expected[key] = row
    for key, row in expected.items():
        source_cache = source_dir / 'pair_cache'
        saved = _strict_load(source_cache / (key + '.json'))
        if saved['pair_key'] != key:
            raise ValueError('Reused cache key mismatch')
        _validate_saved(saved['result'], row, prompt_sha256)
        for number, attempt in enumerate(saved['attempts'], start=1):
            entry = _strict_load(source_cache / (key + '.attempt_' + str(number) + '.json'))
            if entry['attempt'] != attempt or attempt['pair_key'] != key or attempt['attempt'] != number:
                raise ValueError('Reused attempt history mismatch')
            _validate_saved(entry['result'], row, prompt_sha256)
            target = cache_dir / (key + '.attempt_' + str(number) + '.json')
            if not target.exists():
                _atomic_json(target, entry)
        target = cache_dir / (key + '.json')
        if not target.exists():
            _atomic_json(target, saved)
    _atomic_json(marker, dict(source=provenance, unique_pairs=len(expected), created_at=_now()))


def _transient(error):
    if not error:
        return False
    status = error.get('http_status')
    return (
        status == 429
        or isinstance(status, int)
        and 500 <= status < 600
        or error.get('type')
        in {'TimeoutError', 'URLError', 'ConnectionResetError', 'ConnectionAbortedError', 'RemoteDisconnected'}
    )


def _validate_saved(result, row, prompt_sha256):
    if result['source_text'] != row['source_text'] or result['asr_transcript'] != row['asr_transcript']:
        raise ValueError('Saved judgment texts do not match their input record')
    if result['audit']['prompt_sha256'] != prompt_sha256:
        raise ValueError('Saved judgment uses a different prompt')
    if row.get('input_error') is None:
        summarize([result])
    elif result['judgment'] is not None or result['score_1_to_5'] is not None:
        raise ValueError('Invalid input must remain unscored')


def _record_metadata(row):
    return {
        key: row[key]
        for key in ('record_key', 'model', 'label', 'repeat', 'id', 'metrics_row', 'metrics_path', 'source_origin')
    }


def _read_rows(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8-sig')
    if text.lstrip().startswith('['):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError('Expected a JSON array or JSONL objects: ' + str(path))
    return rows


def _index_unique(index, key, row):
    if key in index and index[key] != row:
        raise ValueError('Ambiguous source-manifest identity; no row-order or fuzzy matching is allowed')
    index[key] = row


def _audio_key(value, parent):
    path = Path(value)
    return str((path if path.is_absolute() else parent / path).resolve())


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode('utf-8')).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _strict_load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=reject_duplicate_keys)


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _append_json(path, value):
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=True) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def _read_journal(path):
    if not path.exists():
        return {}
    rows = {}
    with open(path, 'rb+') as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            # A killed append may leave an incomplete final line. Completed
            # judgments remain in the atomic pair cache and are reused.
            if not line.endswith(b'\n'):
                handle.seek(start)
                handle.truncate()
                break
            row = json.loads(line, object_pairs_hook=reject_duplicate_keys)
            key = row['record_key']
            if key in rows:
                raise ValueError('Duplicate final judgment in output journal')
            rows[key] = row
    return rows


@contextmanager
def _output_lock(output_dir):
    with open(output_dir / '.lock', 'a+b') as handle:
        if os.name == 'nt':
            import msvcrt

            if handle.tell() == 0:
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_table(path, results):
    lines = ['Model\tN\tMean score (1-5)\tScore 5 (% all)\tCoverage (%)\tReviews\tErrors']
    for name, summary in results.items():
        mean = summary['mean_score_1_to_5']
        lines.append(
            '\t'.join(
                map(
                    str,
                    [
                        summary.get('label', name),
                        summary['total'],
                        None if mean is None else round(mean, 4),
                        summary['percent_score_5_of_all_cases'],
                        summary['scored_coverage_percent'],
                        summary['judge_review_count'],
                        summary['request_or_validation_error_count'],
                    ],
                )
            )
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
