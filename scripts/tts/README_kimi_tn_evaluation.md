# Kimi semantic evaluation of TTS normalization

`evaluate_tts_llm.py` scores the saved ASR transcript of generated audio against
its original raw written input. It runs after the existing TTS/ASR evaluation;
it does not generate speech, rerun ASR, or require a GPU. Python 3.10+ and the
standard library are sufficient. The script reads `NVIDIA_API_KEY` (or `API_KEY`)
from the environment; never place the key in a command argument or input file.

```bash
python scripts/tts/evaluate_tts_llm.py \
  --metrics /absolute/path/Singleturn_en_TN_filewise_metrics_0.json \
  --model-name replication_158k \
  --reference-manifest /absolute/path/eval_singleturn.json \
  --output-dir /absolute/path/kimi_scores \
  --workers 64 --retry-invalid --max-attempts 10
```

For multiple models, pass `--inputs models.json` instead of `--metrics`. Each
model can list multiple completed repeats; the output also reports each repeat:

```json
{
  "replication_158k": {
    "label": "V1.1 replication, 158k",
    "metrics": ["/absolute/path/filewise_metrics_0.json"],
    "reference_manifest": "/absolute/path/eval_singleturn.json"
  },
  "frontend_delay8": {
    "label": "V1.1 replication, 158k + delay-8 frontend",
    "metrics": "/absolute/path/frontend_filewise_metrics_0.json",
    "reference_manifest": "/absolute/path/eval_singleturn.json"
  }
}
```

Input JSON arrays and JSONL are accepted. The ASR field defaults to `pred_text`
and is passed unchanged, including empty strings. Missing/non-string transcripts
remain unscored and are counted. This does not undo any text cleaning already
applied when the original evaluation JSON was written.

Raw source is recovered from the reference manifest by stable ID or the exact
resolved `gt_audio_filepath`, never row order or fuzzy text matching. This is
necessary for `_IPATextInput` runs, whose `tts_text_input` contains predicted IPA.
Without a manifest, the runner checks `source_text`, `raw_text`, `original_text`,
then non-IPA `tts_text_input`. It never automatically uses `gt_text`, normalized
reference text, reference-audio ASR, or IPA as the source. Unresolvable raw inputs
remain unscored. `--source-field` and `--asr-field` support other explicit schemas.
The automatic reference join expects scalar single-turn source records. Multiturn
filewise metrics can be scored directly when they preserve raw `tts_text_input`;
otherwise supply a canonical per-turn JSONL with `source_text` and `pred_text`.

To evaluate the dataset normalizer itself, use a JSON/JSONL containing raw
`source_text` and the dataset `normalized_text`, and pass
`--source-field source_text --asr-field normalized_text`. This sends the exact
normalized text as the candidate instead of an ASR transcript. Report this
separately as a reference-text baseline without TTS/ASR, rather than a measured
speech-generation result or a guaranteed upper bound.

The judge receives only the unchanged full system prompt and the original two
strings `source_text` and `asr_transcript`. Model names, source IDs, normalized
references, categories, and earlier scores are kept outside its request.
The supplied context-only prompt is preserved byte for byte (SHA-256
`14ef7d6b042d83cf125742bac5b29ba7e452d74f4210479a7174081db0c98dc3`).
The request uses `nvidia/moonshotai/kimi-k3` at NVIDIA's inference endpoint,
with temperature 0, top_p 1, reasoning_effort low, max_tokens 8192, and no streaming.
It permits contextually equivalent verbalizations and checks material quantities,
units, dates, identifiers, omissions, additions, and full-sentence meaning.

The primary result is an ordinal **1–5 semantic-preservation score**:

- 5: complete meaning preserved, including equivalent wording/readings.
- 4: minor nonessential meaning loss; central facts/entities remain intact.
- 3: substantive error or omission with important content still preserved.
- 2: major corruption with some relevant content remaining.
- 1: meaning absent, unrelated, or centrally reversed.

A valid `REVIEW` judgment has a null score. HTTP failures, truncated output,
invalid JSON, inconsistent grades, and nonliteral evidence also remain unscored.
They are not converted to a low TTS score. The script reports the mean on scored
cases, score histogram, percentage of **all** cases receiving 5, scored coverage,
review counts, error counts, and entity-check counts. A 1–5 score is not an accuracy
percentage. This is an ASR-mediated text metric: it does not assess voice/prosody,
and a mismatch can originate from either ASR or TTS. Judge reliability still needs
manual checks; the structural validator cannot establish semantic correctness.

Results are saved incrementally:

- `judgments.jsonl`: one final row per declared model/repeat/sample, including errors.
- `summary.json` and `summary.txt`: aggregate scores and coverage for each model.
- `input_records.json` and `run.json`: resolved inputs, hashes, prompt/settings provenance.
- `pair_cache/`: atomic judgments and every API attempt, with final content and usage.
- `attempts.jsonl`: consolidated attempt metadata, containing no credentials.

Identical raw-source/ASR pairs share one judgment across models and repeats. Every
original sample still contributes once to its own model summary. The request is
blind to the TTS identity. `same_source_cohort_first_repeat` verifies whether the
first repeats contain the same raw-source multiset; unmatched cohorts are not
silently intersected or discarded.

Repeat the same command to resume. Completed judgments are reused; an interrupted
last JSONL append is repaired using atomic cached results. Changed inputs, code,
or prompt require a new output directory. Each attempt is persisted before any
retry, so a restart does not resample a valid judgment or reset the retry budget.
By default only 429, 5xx, and transport failures receive bounded exponential
backoff. Opt in with `--retry-invalid` to retry malformed JSON, truncated output,
inconsistent grades, or nonliteral evidence. The **first valid judgment is final**,
including a low score or a valid REVIEW; neither is resampled. Every failed response
and every retry is retained in `pair_cache/`, with its original content and audit.
The original prompt, input strings, and request parameters stay unchanged. Defaults
are four concurrent requests, four total attempts, and a 180-second request timeout.
Use `--workers`, `--max-attempts`, and `--timeout` to adjust these limits. The attempt
cap is per unique pair, includes the original attempt, and survives restarts.
Changing the retry policy or cap requires a new output directory.

To recover invalid judgments from an already completed run without rejudging its
valid scores or REVIEWs, use a separate output directory and `--reuse-results`:

```bash
python scripts/tts/evaluate_tts_llm.py --inputs models.json \
  --output-dir kimi_scores_retry --reuse-results kimi_scores \
  --retry-invalid --max-attempts 10 --workers 64
```

Reuse requires identical source files, model specifications, prompt, API settings,
client, and validator. The source results remain untouched; their fingerprint is
recorded in the new run. Original and retry attempt files are retained together.
Only invalid responses with remaining attempt budget trigger new calls. Exhausted
or nonretryable errors remain unscored and counted. A valid REVIEW describes judge
uncertainty rather than a malformed response and remains unscored. API validity
does not guarantee a semantically correct grade; manual review is still needed.

The client, consistency validator, and prompt originate from the user-supplied
`kimi_tts_asr_evaluation.zip`. The integration retains that protocol and adds safe
raw-source resolution, checkpointed batching, and audit data for malformed output.

Benchmark API concurrency in a separate output directory:

```bash
python scripts/tts/evaluate_tts_llm.py --inputs models.json --output-dir timing_trial --benchmark-workers 1 4 8 16 --benchmark-samples 16
```

Each timing trial submits the same seeded sample pairs with one request per pair
(no retries). Timing judgments are saved separately and never selected for the
model-score table. `benchmark_summary.json` reports throughput, elapsed time,
speedup, errors and a recommended worker count from error-free trials. This is a
small timing trial under observed server load, not a general throughput guarantee.
