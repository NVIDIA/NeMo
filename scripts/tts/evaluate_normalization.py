# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare full-pipeline normalization using saved TTS ASR filewise metrics.

Example:
    python scripts/tts/evaluate_normalization.py --inputs comparison.json \
        --reference-manifest /data/unseen/eval.json \
        --lexicon examples/tts/conf/magpietts/phoneme_loss_lexicon_en.yaml \
        --base-path /experiments --output-dir /reports/normalization

The input JSON maps model IDs to {"label": ..., "metrics": ...}. Multiturn
sidecars are discovered next to each gt_audio_filepath, or supplied explicitly
as "turn_manifest". Match by audio path then source_sample_idx/turn_id, never
by error-sorted metric row order. Canonical JSONL records with id/raw_text/
reference_text/hypothesis_text are also supported with "format": "canonical".
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path

import yaml

from nemo.collections.tts.parts.utils.text_normalization_metrics import (
    aggregate_text_scores,
    build_lexicon_index,
    score_text_normalization,
    tokenize_words,
)


def load_model_records(spec, base_path, source_records):
    """Load one run, validate reference identity, and return stable IDs plus input hashes."""
    path = _resolve(spec['metrics'], base_path)
    rows = _read_rows(path)
    provenance = dict(metrics_path=str(path), metrics_sha256=_sha256(path), metric_rows=len(rows), sidecars=[])
    records, manifests = {}, {}
    if spec.get('format') == 'canonical':
        for row in rows:
            identifier = row['id']
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f'Canonical turn ID must be a nonempty string: {identifier!r}')
            if identifier in records:
                raise ValueError(f'Duplicate turn ID {identifier} in {path}')
            for key in ['raw_text', 'reference_text', 'hypothesis_text']:
                if not isinstance(row.get(key), str):
                    raise ValueError(f'{identifier}: {key} must be a string')
            records[identifier] = dict(row)
        return records, provenance
    source_by_id = {row.get('id', row.get('cut_id')): row for row in source_records}
    for index, row in enumerate(rows):
        if not isinstance(row.get('pred_text'), str):
            raise ValueError(f'{path} row {index}: pred_text must be a string; missing ASR cannot be skipped')
        audio = Path(row['gt_audio_filepath'])
        if not audio.is_absolute():
            audio = path.parent / audio
        explicit = spec.get('turn_manifest')
        manifest = (
            _resolve(explicit, base_path) if explicit else audio.parent / 'multiturn_user_audio_turn_manifest.jsonl'
        )
        sidecar = None
        if manifest.exists():
            if manifest not in manifests:
                loaded = {}
                for item in _read_rows(manifest):
                    target = os.path.normpath(str(manifest.parent / item['audio_filepath']))
                    if target in loaded:
                        raise ValueError(f'Duplicate target audio {target} in {manifest}')
                    loaded[target] = item
                manifests[manifest] = loaded
                provenance['sidecars'].append(dict(path=str(manifest), sha256=_sha256(manifest)))
            sidecar = manifests[manifest].get(os.path.normpath(str(audio)))
            if sidecar is None:
                raise ValueError(f'{audio} not found in sidecar {manifest}')
            source_index = sidecar['source_sample_idx']
            if type(source_index) is not int or not 0 <= source_index < len(source_records):
                raise ValueError(f'{manifest}: invalid source_sample_idx {source_index!r}')
            source = source_records[source_index]
            turn = sidecar.get('turn_id', 0)
            if isinstance(source['text'], list):
                if type(turn) is not int or not 0 <= turn < len(source['text']):
                    raise ValueError(f'{manifest}: invalid turn_id {turn!r} for source index {source_index}')
                raw = source['text'][turn]
                reference = source['normalized_text'][turn]
                conversation_id = source.get('cut_id', source.get('id', str(source_index)))
                identifier = f'{conversation_id}-turn_{turn:03d}_agent'
            else:
                raw, reference = source['text'], source['normalized_text']
                conversation_id = source.get('id', source.get('cut_id', str(source_index)))
                identifier = conversation_id
            if _identity_text(sidecar['text']) != _identity_text(raw):
                raise ValueError(f'{path} source index {source_index}, turn {turn}: raw reference mismatch')
        else:
            # Single-turn adapter: explicit stable IDs, not an assumed row-order mapping.
            identifier = row.get(spec.get('id_field', 'id'))
            if identifier not in source_by_id:
                raise ValueError(f'No stable source identity for {audio}; supply a turn_manifest or id_field')
            source = source_by_id[identifier]
            raw, reference = source['text'], source['normalized_text']
            conversation_id, turn = source.get('source_conversation_id', identifier), source.get(
                'source_turn_index', 0
            )
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f'{identifier}: source normalized_text is empty or not a string')
        if identifier in records:
            raise ValueError(f'Duplicate source turn {identifier} in {path}; choose one repeat per model')
        text_input = row.get('tts_text_input')
        if text_input is not None and _identity_text(text_input) != _identity_text(raw):
            raise ValueError(f'{identifier}: saved TTS input differs from source raw text')
        records[identifier] = dict(
            id=identifier,
            conversation_id=conversation_id,
            turn_index=turn,
            raw_text=raw,
            reference_text=reference,
            hypothesis_text=row['pred_text'],
            reference_audio_asr=row.get('gt_audio_text'),
            stored_reference=row.get('gt_text'),
            stored_cer=row.get('cer'),
            stored_wer=row.get('wer'),
            gt_audio_filepath=str(audio),
            pred_audio_filepath=row.get('pred_audio_filepath'),
            metrics_row_index=index,
            frontend_reached_eos=sidecar.get('frontend_reached_eos') if sidecar else None,
            frontend_decoder_steps=sidecar.get('frontend_decoder_steps') if sidecar else None,
        )
    return records, provenance


def compare_models(models, lexicon_index, cohort='strict'):
    """Score identical turns/reference text across models; make missing coverage explicit."""
    if not models:
        raise ValueError('At least one model is required')
    common = set.intersection(*(set(rows) for rows in models.values()))
    union = set.union(*(set(rows) for rows in models.values()))
    if not common:
        raise ValueError('No common turn IDs')
    if cohort == 'strict' and common != union:
        raise ValueError(
            f'Turn coverage differs: {len(common)} common / {len(union)} union; '
            'use --cohort intersection only after reviewing missing turns'
        )
    scored, comparison = {}, []
    for identifier in sorted(common):
        references = {(rows[identifier]['raw_text'], rows[identifier]['reference_text']) for rows in models.values()}
        if len(references) != 1:
            raise ValueError(f'Inconsistent raw/reference text for common ID {identifier}')
        row_scores = {}
        for name, rows in models.items():
            record = rows[identifier]
            score = score_text_normalization(
                record['raw_text'], record['reference_text'], record['hypothesis_text'], lexicon_index
            )
            row_scores[name] = dict(record, **score)
            scored.setdefault(name, []).append(row_scores[name])
        comparison.append(dict(id=identifier, models=row_scores))
    summary = {}
    for name, rows in scored.items():
        values = aggregate_text_scores(rows)
        values.update(
            available_turns=len(models[name]),
            excluded_turn_ids=sorted(set(models[name]) - common),
            legacy_mean_cer=_finite_mean(row.get('stored_cer') for row in rows),
            legacy_mean_wer=_finite_mean(row.get('stored_wer') for row in rows),
            stored_reference_word_mismatches=sum(
                row.get('stored_reference') is not None
                and _word_keys(row['stored_reference']) != _word_keys(row['reference_text'])
                for row in rows
            ),
            frontend_truncated_turns=sum(row.get('frontend_reached_eos') is False for row in rows),
        )
        by_turn = defaultdict(list)
        for row in rows:
            by_turn[row.get('turn_index', 0)].append(row)
        values['by_turn'] = {str(turn): aggregate_text_scores(items) for turn, items in sorted(by_turn.items())}
        summary[name] = values
    return dict(common_turns=len(common), union_turns=len(union), cohort=cohort, models=summary), comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--inputs', type=Path, required=True, help='Model ID -> label/metrics/optional turn_manifest JSON'
    )
    parser.add_argument('--reference-manifest', type=Path, help='Dataset JSONL with raw text and normalized_text')
    parser.add_argument(
        '--lexicon', type=Path, required=True, help='phoneme_loss_lexicon_en.yaml; words only are used'
    )
    parser.add_argument(
        '--base-path', type=Path, default=Path('.'), help='Root for relative input metric/sidecar paths'
    )
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cohort', choices=['strict', 'intersection'], default='strict')
    args = parser.parse_args()
    specs = json.loads(args.inputs.read_text(encoding='utf-8'))
    if any(spec.get('format') != 'canonical' for spec in specs.values()) and not args.reference_manifest:
        parser.error('--reference-manifest is required for NeMo filewise metric inputs')
    sources = _read_rows(args.reference_manifest) if args.reference_manifest else []
    lexicon = yaml.safe_load(args.lexicon.read_text(encoding='utf-8'))
    lexicon_index = build_lexicon_index(lexicon['entries'])
    models, provenance = {}, {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = args.output_dir / 'canonical_inputs'
    snapshots.mkdir(exist_ok=True)
    for model, spec in specs.items():
        if not model or Path(model).name != model or model in {'.', '..'}:
            raise ValueError(f'Model ID must be a safe filename component: {model!r}')
        models[model], provenance[model] = load_model_records(spec, args.base_path, sources)
        snapshot = snapshots / f'{model}.jsonl'
        _write_jsonl(snapshot, (models[model][key] for key in sorted(models[model])))
        provenance[model]['canonical_snapshot_sha256'] = _sha256(snapshot)
    summary, comparison = compare_models(models, lexicon_index, args.cohort)
    summary['labels'] = {key: spec.get('label', key) for key, spec in specs.items()}
    summary['definitions'] = dict(
        reference='Dataset normalized_text shared by every model; stored gt_text is audited, not trusted as common GT.',
        transcription='Saved pred_text from ASR on generated speech; no ASR rerun or reference substitution.',
        normalization='NFKC, casefold, Unicode word extraction, curly apostrophe folding; no number re-normalization.',
        cer='Pooled character edits / reference characters; canonical words joined by one space.',
        wer='Pooled word edits / reference words.',
        normalization_group_exact='Exact complete reference lexicon groups, averaged within eligible turns, then over turns.',
        normalization_group_cer='Character CER within complete groups, averaged within eligible turns, then over turns.',
        grouping='Consecutive reference lexicon words, whitespace/hyphens or one and connector; punctuation breaks groups; '
        'only runs containing raw-to-normalized changed words are scored.',
        insertion_policy='Charge each insertion adjacent to a group once, preferring right then left; minimum-edit '
        'word alignment ties prefer diagonal, deletion, insertion.',
        missing='Empty ASR strings are deletion errors; missing records are not silently dropped; strict common cohort.',
        limitations='ASR-dependent, strict wording, no alternative readings accepted; lexicon groups approximate entities '
        'and do not cover all URL/name fragments. Not semantic numeric-value accuracy.',
    )
    from nemo.collections.tts.parts.utils import text_normalization_metrics

    summary['provenance'] = dict(
        inputs_path=str(args.inputs),
        inputs_sha256=_sha256(args.inputs),
        python_version=platform.python_version(),
        reference_manifest=str(args.reference_manifest) if args.reference_manifest else None,
        reference_sha256=_sha256(args.reference_manifest) if args.reference_manifest else None,
        lexicon_path=str(args.lexicon),
        lexicon_sha256=_sha256(args.lexicon),
        lexicon_entries=len(lexicon['entries']),
        script_sha256=_sha256(Path(__file__)),
        scorer_sha256=_sha256(Path(text_normalization_metrics.__file__)),
        models=provenance,
    )
    _write_json(args.output_dir / 'summary.json', summary)
    _write_jsonl(args.output_dir / 'turn_scores.jsonl', comparison)
    _write_tables(args.output_dir, summary)
    print((args.output_dir / 'comparison.md').read_text(), flush=True)
    print(f'Saved {len(comparison)} matched turns, all group spans, and hashed canonical inputs in {args.output_dir}')


def _read_rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        first = next((character for character in iter(lambda: handle.read(1), '') if not character.isspace()), '')
        handle.seek(0)
        if first == '[':
            rows = json.load(handle)
        else:
            rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f'Expected nonempty JSON array or JSONL records: {path}')
    return rows


def _resolve(path, base):
    path = Path(path)
    return path if path.is_absolute() else base / path


def _identity_text(text):
    return ' '.join(unicodedata.normalize('NFKC', text).replace('’', "'").split())


def _word_keys(text):
    return [word['key'] for word in tokenize_words(text)]


def _finite_mean(values):
    values = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return statistics.fmean(values) if values else None


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def _write_tables(directory, summary):
    columns = [
        'model',
        'turns',
        'eligible_turns',
        'groups',
        'cer',
        'wer',
        'normalization_group_exact',
        'normalization_group_cer',
        'whole_turn_exact',
        'empty_hypotheses',
    ]
    with (directory / 'comparison.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for model, values in summary['models'].items():
            writer.writerow(dict(model=summary['labels'][model], **{key: values[key] for key in columns[1:]}))
    lines = [
        'All rates are percentages. CER/WER pool errors across turns. Group rates average equally over eligible turns.',
        '',
        '| Model | Turns | Group turns | CER ↓ | WER ↓ | Group exact ↑ | Group CER ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for model, values in summary['models'].items():
        rates = [
            '—' if values[key] is None else f'{100 * values[key]:.2f}%'
            for key in ['cer', 'wer', 'normalization_group_exact', 'normalization_group_cer']
        ]
        lines.append(
            f"| {summary['labels'][model]} | {values['turns']} | {values['eligible_turns']} | "
            + ' | '.join(rates)
            + ' |'
        )
    (directory / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
