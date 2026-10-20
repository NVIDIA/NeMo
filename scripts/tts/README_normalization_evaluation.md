# Full-pipeline text-normalization evaluation

`evaluate_normalization.py` compares the ASR transcripts of generated TTS audio
against a shared dataset reference. It reuses saved filewise metrics and does not
run TTS, ASR, a phonemizer, or a text-normalization model. The implementation is in
`nemo/collections/tts/parts/utils/text_normalization_metrics.py`.

## Metrics

- **CER/WER**: pooled character/word edit counts divided by pooled reference
  lengths over all matched turns, including turns without normalization groups.
  NFKC, case folding, Unicode word extraction and curly-apostrophe folding are
  applied equally to the source reference and saved `pred_text`; words are joined
  with one space for CER. Digits are not converted to words again.
- **Normalization-group exact accuracy**: find reference words/phrases in
  `phoneme_loss_lexicon_en.yaml`, join consecutive matches across whitespace or
  hyphens (or a single `and` connector), and retain groups containing words changed
  from raw text. Raw-to-normalized changes use `SequenceMatcher(autojunk=False)`.
  Punctuation breaks groups. Compare each complete reference group against its
  ASR span using minimum-edit full-sentence word alignment. A group is correct
  only if it has zero substitutions, deletions or insertions. Average the fraction
  of correct groups within each eligible turn, then average equally over turns.
- **Normalization-group CER**: character edit distance between each complete
  reference group and its projected ASR span, divided by reference group length;
  average within eligible turns, then equally over turns. This gives partial
  credit while exact accuracy detects incorrect complete numbers.

Only the YAML's orthographic words are used; IPA pronunciations do not participate.
This follows the earlier phoneme-group scorer's grouping, reference-only selection,
macro averaging, and insertion-attribution policy. Word alignment breaks equal-cost
ties in diagonal/deletion/insertion order. Insertions inside or beside groups are
charged once, preferring the right group, otherwise the left group.

The report also includes pooled group accuracy/CER, whole-turn exact accuracy,
all-groups-correct turn accuracy, coverage, empty transcripts, frontend truncation
flags (when available), and a breakdown by conversation turn position. Empty ASR
strings count as deletion errors. Hallucinated output can produce CER/WER above
100%; rates are not clipped. Turns with no groups remain in overall CER/WER but
do not enter the group-accuracy denominator.

These metrics depend on ASR and strict reference wording. For example, an optional
`and` can fail exact matching. The code does not globally remove `and`, accept
alternative readings, collapse homophones, or claim semantic numeric-value
accuracy. Lexicon groups approximate entities: an open-vocabulary URL or name can
be only partly covered. Changed-word coverage and all scored spans are saved for
review. Use listening checks to distinguish ASR errors from speech errors.

## Reproducible comparison

From the repository root, create an input JSON mapping model IDs to filewise
metric files and readable labels. Paths may be absolute or relative to `--base-path`:

```json
{
  "candidate": {
    "label": "Candidate",
    "metrics": "candidate/dataset_filewise_metrics_0.json"
  },
  "frontend": {
    "label": "Speech model + normalizer frontend",
    "metrics": "frontend/dataset_filewise_metrics_0.json"
  }
}
```

```bash
python scripts/tts/evaluate_normalization.py \
  --inputs /reports/models.json \
  --reference-manifest /data/unseen/eval.json \
  --lexicon examples/tts/conf/magpietts/phoneme_loss_lexicon_en.yaml \
  --base-path /experiments \
  --output-dir /reports/full_pipeline_normalization
```

The reference manifest must contain `text` and `normalized_text`. Multiturn entries
use parallel lists and a stable `cut_id` (or `id`). The loader discovers
`multiturn_user_audio_turn_manifest.jsonl` beside each `gt_audio_filepath`, joins
the metric to the sidecar **by audio path**, then uses `source_sample_idx` and
`turn_id` to look up the original dataset turn. An explicit `turn_manifest` path
can be supplied per model. Error-sorted metric row order is never a source ID.
The sidecar and saved `tts_text_input` must match the dataset's raw text.

Every model is scored against the same dataset `normalized_text`. Existing
`gt_text` is audited because older evaluation versions may have regenerated it
with a text normalizer. Existing filewise CER/WER averages are retained as legacy
diagnostics; they can differ from the common-reference scores in the table.

The default `--cohort strict` requires identical turn IDs and references across
all models. `--cohort intersection` explicitly permits comparing common turns;
all excluded IDs are then reported. Supply one evaluation repeat per model.

Single-turn filewise records with explicit stable IDs can use `id_field` to join
a scalar reference manifest. Other adapters can produce canonical JSONL records
and set `"format": "canonical"` per model:

```json
{"id":"sample-1","raw_text":"Pay $125.","reference_text":"Pay one hundred twenty five dollars.","hypothesis_text":"pay one hundred twenty four dollars"}
```

For canonical inputs, `--reference-manifest` is optional, and records with the
same ID must still have identical raw/reference text across models.

Outputs:

- `summary.json`: counts, metrics, definitions, per-turn-position breakdown and
  SHA-256 provenance for inputs, source reference, lexicon and scorer code.
- `comparison.md` and `comparison.csv`: the main comparison table (CSV rates are
  fractions, Markdown rates are percentages).
- `turn_scores.jsonl`: every scored sentence and complete GT/generated group span,
  counts and coverage, suitable for manual review and later category analysis.
- `canonical_inputs/`: deterministic standalone per-model inputs, hashed in the
  summary and reusable with `format: canonical`.

No training or inference defaults are changed by this standalone scorer.
