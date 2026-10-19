# Alignment-derived phoneme waits for text-only pretraining

Enable in `easy_magpietts_lhotse.yaml` or on the training command line:

```bash
model.dynamic_phoneme_delay=true
```

The default is `false`. Disabled runs use the existing fixed-delay collation and
model path. When enabled, only `task: text_only` batches change; speech batches
retain their original audio/text/phoneme timing.

Dynamic text-only mode ignores `streaming_phonemes_delay`: its fixed phoneme
offset is **zero** in both training and the text frontend. The conditioning
prefix is retained. All additional lookahead comes from alignment-derived or
predicted PAD/wait tokens.

Requirements:

- `use_multiturn_dataset=true`, `use_lhotse=true`, an IPA tokenizer with a PAD ID,
  and `phoneme_stacking_factor=1`.
- All training modes use streaming text. Their configured fixed phoneme delays
  are ignored for dynamic text-only batches and retained for speech batches.
- Each `txt_norm_jsonl` input declares `duration_text_tokenizer_path` and
  `duration_phoneme_tokenizer_path`, matching the model tokenizers. Duration
  batching automatically reserves a conservative source-plus-IPA length bound
  when enabled. Keep `duration_padding_tokens` large enough for the existing
  conditioning prefix. Batch occupancy can decrease with this bound.
- Prefer the supplied version-1.x `word_alignment` metadata: Unicode code-point,
  zero-based/end-exclusive ranges; original-string SHA256 hashes; complete
  `groups`; and source/normalized/IPA word mappings.

The CPU loader reuses text and phoneme collation, including its raw/normalized
transcript choice. It maps the complete strings with tokenizer offsets, projects
source offsets through timestamp removal/whitespace collapse, and checks every ordinary word and shared phrase as well as normalization
expressions. It gates each IPA span until the whole source expression and its closing
boundary (or text EOS) are visible. An IPA BPE crossing a span boundary is gated
as a whole. When normalized text is selected, its individual normalized words
provide the dependencies. All earlier waits count toward visibility; no fixed
phoneme delay is added.

Dynamic text-only batches compact whitespace runs in the selected text and IPA
to one word separator and remove leading/trailing whitespace before tokenizing.
Empty or whitespace-only IPA alignment groups do not create wait dependencies.
All non-whitespace characters are retained. Alignment hashes and character spans
are checked against the original strings, with offsets projected through the
cleanup; manifests and their alignment metadata are not rewritten. Disabled
runs and speech batches keep their existing whitespace handling.

Waits reuse the existing phoneme PAD ID **inside the true target length**. They
are supervised by cross-entropy, have a zero input embedding, and receive unit
loss weight even when lexicon amplification is enabled. Right/batch padding
remains outside the loss. Lexicon masks move with their original phonemes.
The existing validation phoneme-token accuracy excludes PAD/wait targets.

Shared engine IPA spans are usable without invented internal word boundaries.
The reader resolves connected source/normalized/shared-IPA groups, including
shared phrases crossing normalization-group boundaries. Ordinary words in a
long unchanged group keep individual dependencies. Normalized input uses its
own word/shared-phrase dependencies rather than a complete raw-number expansion.
Checking a word does not automatically insert waits: all earlier waits and IPA
tokens already advance the text clock, and only a remaining readiness gap adds PADs.

Only `txt_norm_jsonl` readers with dynamic mode enabled filter unusable records,
before bucketing and distributed batching. Missing/stale hashes, incomplete
spoken-IPA coverage, genuinely missing boundaries, invalid ranges, or an explicit
`dynamic_delay_usable: false` cause exclusion. Shared `partial` records remain
usable when their complete dependencies resolve. Existing records need not carry
a usability flag; a true flag never bypasses validation. Raw and normalized
mappings must both resolve so transcript selection cannot change eligibility.
Readers log excluded/seen counts per manifest and fail clearly if an entire pass
has no usable records. Validation uses the same eligibility rules: report these
exclusion counts alongside metrics. Original manifests and phoneme strings are
never rewritten, and disabled readers keep their existing record selection.

No whole-text-EOS fallback is supervised. A custom cut bypassing the filtering
reader with unusable metadata raises a descriptive error before model execution.

Training logs aggregate across all ranks, including mixed speech/text batches:

- `train/dynamic_phoneme_wait_tokens`: inserted waits per global microbatch.
- `train/dynamic_phoneme_wait_percent`: waits / valid phoneme-channel tokens,
  including BOS/EOS in the denominator.
- `train/dynamic_phoneme_fallback_percent`: retained for log compatibility; zero
  for accepted turns. Excluded records are reported by the reader, not this metric.

`TextOnlyPhonemeFrontend` reads the checkpoint's saved flag. PAD predictions
advance the text/decoder clock with zero phoneme input and are omitted from the
returned IPA/token IDs. No reference alignment is needed at inference. Its
`wait_tokens` output reports generated waits. `--phoneme_frontend_expected_delay`
guards the trained delay for fixed-delay checkpoints; dynamic checkpoints ignore
this argument and always use zero fixed phoneme delay. Its `first_phoneme_step`
reports the one-based step of the first non-PAD/BOS/EOS prediction, excluding
the conditioning prefix and including any fixed delay. Frontend metadata reports
the effective `phoneme_delay: 0` and the ignored `configured_phoneme_delay`.

This feature does not shift audio targets or implement dynamic-delay joint speech
generation. In the two-model cascade, the frontend finishes IPA generation before
the existing TTS model consumes the resulting phonemes.

### Validation timing and wait coverage

Text-only validation always logs the loader's target schedule under
`val/text_only_target_*`. This is alignment-derived supervision, not a measure
of the model's autoregressive waiting behavior. Existing token accuracy remains
teacher-forced and excludes PAD/wait targets.

The EasyMagpie Lhotse configs now run actual greedy autoregressive validation
on every eligible text-only sequence, at every validation interval. No sample
budget overrides are required. Both settings default to `null`:

```yaml
model:
  text_only_val_ar_max_samples: null
  text_only_val_ar_max_steps: null
```

`text_only_val_ar_max_samples: null` evaluates all eligible sequences across all
validation batches on each rank. A positive integer retains optional sampling
(per rank per validation interval), and `0` disables only the AR diagnostic.
Older restored configs that lack this key retain the previous disabled behavior;
set `++model.text_only_val_ar_max_samples=null` to opt them into full evaluation.

`text_only_val_ar_max_steps: null` uses the frontend's existing per-sequence cap,
`max(256, 8 * text_decoder_steps + 128)`, instead of a fixed 2,048-step ceiling.
A positive integer optionally imposes a smaller ceiling. This cap is still
required to stop a model that produces PADs forever or never reaches EOS.
Truncated predictions remain visible through EOS completion and no-phoneme rates.

Autoregressive validation performs an additional cached forward for each
generated step, so full evaluation costs more than teacher-forced validation.
Generation skips Lightning sanity checks, loads no extra model or ASR, and never
calls the audio decoder or consumes reference IPA. It reuses the loader's
selected text/context token IDs, the same cached decoder as the standalone
frontend, and validation precision. Streaming mode and phoneme stacking factor
one are required only when actually evaluating a text-only batch; audio-only
runs are unaffected. Positive explicit sample budgets retain early architecture
validation.

Both `val/text_only_target_*` and `val/text_only_ar_*` expose:

| Suffix | Meaning |
| --- | --- |
| `first_phoneme_step` | Mean one-based decoder step of the first real phoneme. Excludes context/prefix prefill, includes fixed delay and earlier generated PAD/BOS predictions. An immediate phoneme is step 1; steps before it equal this value minus 1. |
| `first_phoneme_step_max` | Maximum first-phoneme step across all ranks. |
| `wait_percent` | 100 × PAD / (PAD + real phonemes), within each sequence's valid length. BOS/EOS and batch padding are excluded. |
| `wait_percent_mean` | Mean of individual sequences' wait percentages, so long sequences do not dominate. |
| `no_phoneme_percent` | Percentage of sequences producing no real phoneme. These never contribute artificial zero latency to the mean/max; latency metrics are omitted if none produced a phoneme. |
| `eos_percent` | Percentage reaching EOS. For AR decoding, 100 minus this value is the truncation percentage. Truncated predictions remain in wait coverage and first-phoneme statistics if a phoneme was produced. |
| `samples` | Number of sequences represented. |

Sums/counts and maxima are reduced across all ranks, including ranks receiving
only speech, before ratios are computed. First-phoneme latency and macro coverage
are sequence-weighted; `wait_percent` is token-weighted. Fixed startup delay is
counted in latency but is not an emitted PAD token in coverage.

Target statistics cover every text-only validation sequence, including packed
turns. AR diagnostics currently require single-turn examples; multi-turn cuts
are counted under `val/text_only_ar_skipped_multiturn_samples` rather than
silently treating the first generated EOS as the end of an entire packed cut.
No speech validation, training targets, losses, or sampling behavior are changed.

### W&B decoding examples

When a `WandbLogger` is configured, each validation interval also logs
`val/text_only_decoding_examples`, a table built from the AR decodes already
computed for metrics. It includes the loader's actual selected input text (raw
or normalized), original raw text, tokenizer name, reference and generated IPA,
exact phoneme-token match, target/generated first-phoneme steps, per-sequence
wait percentages, generated steps/waits, EOS/truncation status, and identifiers.
Reference and generated IPA exclude PAD/BOS/EOS. Missing-phoneme delays are null.

`model.text_only_val_wandb_max_examples` defaults to **16 rows globally per
validation interval**. It limits only displayed examples; all eligible samples
still contribute to full validation metrics. Set it to `0` to disable tables.
Rows come from a bounded number of first eligible examples on each rank, then
are interleaved and uploaded once by rank zero at the current trainer global
step. Empty/speech-only ranks participate in gathering, so a table can still be
logged when rank zero saw no text. Each interval creates a new table and clears
the previous buffered rows. W&B-free runs and sanity checks do not collect rows;
no extra generation, checkpoint loading, or audio inference is performed.
