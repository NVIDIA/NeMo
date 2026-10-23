# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi evaluation client: original TTS text + ASR transcript -> semantic score 1-5.

Put this file beside system_prompt.txt and validate_judgment.py. Python stdlib only.
No TTS, ASR, preprocessing or live calls occur on import. No automatic score retries.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .validate_judgment import aggregate_scores, reject_duplicate_keys, validate_judgment

API_URL = 'https://inference-api.nvidia.com/v1/chat/completions'
MODEL = 'nvidia/moonshotai/kimi-k3'


class KimiEvaluator:
    def __init__(self, prompt_dir=None, api_key=None, timeout=180):
        self.prompt_dir = Path(prompt_dir) if prompt_dir is not None else Path(__file__).resolve().parent
        prompt_bytes = (self.prompt_dir / 'system_prompt.txt').read_bytes()
        self.system_prompt = prompt_bytes.decode('utf-8')
        self.prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
        self.api_key = api_key or os.environ.get('NVIDIA_API_KEY') or os.environ.get('API_KEY')
        if not self.api_key:
            raise ValueError('Set NVIDIA_API_KEY or API_KEY, or pass api_key explicitly.')
        self.timeout = timeout

    def evaluate(self, source_text, asr_transcript):
        """Return one validated judgment. HTTP/parse/validation failures raise.

        The API sees only the prompt and these two original strings; no model label,
        language/locale, case rules or reference answer is sent. Do not rewrite inputs.
        """
        if not isinstance(source_text, str) or not isinstance(asr_transcript, str):
            raise TypeError('source_text and asr_transcript must both be strings.')
        pair = {'source_text': source_text, 'asr_transcript': asr_transcript}
        payload = {
            'model': MODEL,
            'messages': [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': json.dumps(pair, ensure_ascii=True)},
            ],
            'temperature': 0,
            'top_p': 1,
            'reasoning_effort': 'low',
            'max_tokens': 8192,
            'stream': False,
        }
        request = Request(
            API_URL,
            data=json.dumps(payload, ensure_ascii=True).encode('utf-8'),
            headers={
                'Authorization': 'Bearer ' + self.api_key,
                'Content-Type': 'application/json',
            },
            method='POST',
        )
        started = time.monotonic()
        with urlopen(request, timeout=self.timeout) as response:
            envelope = _strict_json(response.read().decode('utf-8'))
        audit = {
            'requested_model': MODEL,
            'returned_model': envelope.get('model'),
            'response_id': envelope.get('id'),
            'usage': envelope.get('usage'),
            'elapsed_seconds': time.monotonic() - started,
            'prompt_sha256': self.prompt_sha256,
            'request_payload': payload,
        }
        try:
            choice = envelope['choices'][0]
            audit['finish_reason'] = choice.get('finish_reason')
            content = choice['message'].get('content')
            audit['raw_final_content'] = content
            if choice.get('finish_reason') != 'stop':
                raise ValueError('Incomplete judge response')
            judgment = _parse_content(content)
            validate_judgment(judgment, pair)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise InvalidJudgmentError(audit) from error
        return {
            'source_text': source_text,
            'asr_transcript': asr_transcript,
            'score_1_to_5': judgment['score_1_to_5'],
            'judgment': judgment,
            'error': None,
            'audit': audit,
        }

    def evaluate_safe(self, source_text, asr_transcript):
        """Retain request/validation failures as unresolved records, never score 1.

        For a production runner, log failed attempts and apply bounded transport-only
        retries to 429/5xx/timeouts. Do not retry to improve a low valid judge score.
        """
        try:
            return self.evaluate(source_text, asr_transcript)
        except Exception as error:
            status = error.code if isinstance(error, HTTPError) else None
            return {
                'source_text': source_text,
                'asr_transcript': asr_transcript,
                'score_1_to_5': None,
                'judgment': None,
                # Keep exception type; do not log arbitrary messages containing secrets.
                'error': {'type': type(error).__name__, 'http_status': status},
                'audit': getattr(error, 'audit', {'requested_model': MODEL, 'prompt_sha256': self.prompt_sha256}),
            }


class InvalidJudgmentError(ValueError):
    """Unscored malformed/truncated response with credential-free audit evidence."""

    def __init__(self, audit):
        super().__init__('Invalid or incomplete final judgment; inspect its saved audit')
        self.audit = audit


def summarize(results):
    """One result per original evaluation record, including unresolved results."""
    rows = list(results)
    for row in rows:
        if row['judgment'] is not None:
            pair = {'source_text': row['source_text'], 'asr_transcript': row['asr_transcript']}
            validate_judgment(row['judgment'], pair)
            if row['score_1_to_5'] != row['judgment']['score_1_to_5']:
                raise ValueError('Result score disagrees with its validated judgment.')
        elif row['score_1_to_5'] is not None:
            raise ValueError('An unresolved result must not have a fabricated score.')
    stats = aggregate_scores(row['judgment'] for row in rows)
    to_percent = lambda value: 100 * value if value is not None else None
    return {
        'total': stats['total'],
        'scored': stats['total'] - stats['unresolved'],
        'mean_score_1_to_5': stats['mean_score_1_to_5_on_scored'],
        'score_histogram': stats['score_histogram'],
        'percent_score_5_of_all_cases': to_percent(stats['fully_correct_rate']),
        'scored_coverage_percent': to_percent(stats['scored_coverage']),
        'judge_review_count': stats['judge_review'],
        'request_or_validation_error_count': stats['missing_result'],
        'unresolved_percent': to_percent(stats['unresolved_rate']),
    }


def _strict_json(text):
    return json.loads(text, object_pairs_hook=reject_duplicate_keys)


def _parse_content(content):
    if not isinstance(content, str) or not content.strip():
        raise ValueError('Missing final message content.')
    # Accept only an enclosing JSON fence, never extract JSON from arbitrary prose.
    fenced = re.fullmatch(r'\s*```(?:json)?\s*([\s\S]*?)\s*```\s*', content, re.IGNORECASE)
    return _strict_json(fenced.group(1) if fenced else content)


if __name__ == '__main__':
    raise SystemExit('Import KimiEvaluator and summarize; see ../README_kimi_tn_evaluation.md for usage.')
