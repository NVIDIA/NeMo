# Alignment-derived phoneme waits for text-only pretraining

Enable in `easy_magpietts_lhotse.yaml` or on the training command line:

```bash
model.dynamic_phoneme_delay=true
```

The default is `false`. Disabled runs use the existing fixed-delay collation and
model path. When enabled, only `task: text_only` batches change; speech batches
retain their original audio/text/phoneme timing.

### Optional loader-side raw-to-normalized alignment

```bash
model.dynamic_phoneme_delay=true \
model.dynamic_phoneme_delay_recompute_alignment=true
```

`dynamic_phoneme_delay_recompute_alignment` defaults to `false` and requires
dynamic mode. The model propagates it to training and validation `txt_norm_jsonl`
inputs, including nested/multi-config inputs. No manifest rewrite is needed.

When enabled, the CPU loader recomputes raw-to-normalized word dependencies
using monotonic unchanged-word anchors. Changed words between anchors form one
atomic normalization span. For example, `Pay 12 then 34 now` maps `12` to
`twelve` and `34` to `thirty four`, instead of inheriting a single unresolved
region covering the sentence. Matching ignores case and surrounding punctuation
without modifying the original strings or offsets. Attached en/em dashes can
separate anchors; dates, decimals and ordinary hyphenated expressions stay intact.

Internal repeated anchors must be unique within the remaining word intervals.
For example, the two possible `and` matches in `101 and 2` / `one hundred and
one and two` remain one shared span. Adjacent changed expressions without a
reliable unchanged anchor also remain atomic. Pure insertions and deletions use
the adjacent words as conservative context, so moved/absorbed values such as
`May 1` / `first May` cannot lose their source dependency. This is deterministic text
matching, not a semantic parser or proof of minimal possible latency.

Only the raw-to-normalized mapping is rebuilt. The existing `normalized_words`,
`normalized_to_ipa`, exact shared IPA spans, original-string hashes and offset
conventions remain required and validated. Missing IPA boundaries are never
guessed. Unusable records are still excluded before batching; explicit
`dynamic_delay_usable: false` remains respected. The rebuilt metadata is cached
on the supervision as `word_alignment`, with
`text_alignment_method: loader_word_anchors_v1`; duration bounds are unchanged.

The same actual BPE/IPA tokenizer offsets then determine when a span and its
boundary become visible. Matching an anchor during loading does not require
waiting for that complete anchor at runtime. No fixed phoneme delay is added.
This setting changes training targets only; dynamic frontend inference already
predicts its own PAD waits and needs no additional decoding option.

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

Validation uses the regular batched teacher-forced forward by default. Loss and
`val/text_only_phoneme_token_accuracy` retain their existing definitions; token
accuracy excludes ground-truth PAD/wait, BOS and EOS targets. No autoregressive
decoder loop runs unless explicitly enabled.

The same teacher-forced argmax predictions also produce
`val/text_only_teacher_forced_*` metrics, alongside the loader's target schedule
under `val/text_only_target_*`. These add no decoder forward. Both expose:

| Suffix | Meaning |
| --- | --- |
| `first_phoneme_step` | Mean one-based step of the first non-PAD/BOS/EOS phoneme, excluding the conditioning prefix and including fixed delay. An immediate phoneme is step 1. |
| `first_phoneme_step_max` | Maximum first-phoneme step across all ranks. |
| `wait_percent` | 100 × predicted/target PADs divided by PADs plus real phonemes within valid sequence lengths. BOS/EOS and batch padding are excluded. |
| `wait_percent_mean` | Mean of individual sequences' PAD percentages, so long sequences do not dominate. |
| `no_phoneme_percent` | Percentage with no real phoneme. These never contribute artificial zero latency; latency metrics are omitted if every sequence lacks phonemes. |
| `eos_percent` | Percentage containing an EOS prediction/target inside the valid reference horizon. |
| `samples` | Number of sequences represented. |

Teacher-forced predictions use ground-truth phoneme history, including the
alignment-derived waits. All valid reference steps are scored, including wait
positions; predicted EOS does not stop this forward or truncate these metrics.
Consequently, first-phoneme step is a teacher-forced diagnostic, not free-running
inference latency. EOS coverage is not an autoregressive completion rate.

Counts/sums and maxima are reduced across all ranks before computing ratios,
including ranks receiving only speech. Target and teacher-forced metrics cover
every text-only validation sequence, including packed turns, while excluding
conditioning prefixes and batch padding.

### Optional autoregressive evaluation

Autoregressive validation remains available for explicit post-training
diagnostics. It is disabled in both EasyMagpie Lhotse configs:

```yaml
model:
  text_only_val_ar_max_samples: 0
  text_only_val_ar_max_steps: null
```

Set `++model.text_only_val_ar_max_samples=null` to decode every eligible
single-turn validation sequence, or a positive integer to cap samples per rank
per interval. `0` disables the AR path and its example table. Older checkpoints
without the setting also keep AR disabled. The target and teacher-forced metrics
remain available independently.

With AR enabled, `val/text_only_ar_*` reports the corresponding actual greedy
generation statistics. Its EOS percentage measures completion, and 100 minus
this value is the truncation percentage. Initial delay excludes the context
prefix and includes fixed delay and predicted PAD/BOS steps. No reference IPA is
fed to AR decoding. Streaming mode and phoneme stacking factor one are required.
Packed multi-turn cuts are explicitly counted under
`val/text_only_ar_skipped_multiturn_samples` instead of decoding only turn one.

Eligible examples in each loader batch decode together. Each sample stops at
its own phoneme EOS or safety cap; the batch ends when all samples finish.
Finished cache slots remain allocated until the batch completes. Equal-length
conditioning prefixes share one cache; different prefix lengths use separate
cohorts so artificial padding never changes Mamba states or source timing.
Right-padded source embeddings are zeroed after their valid lengths.

`model.validation_ds.dataset.batch_size` controls the available AR batch size.
`text_only_val_ar_max_steps: null` uses `max(256, 8 * text_decoder_steps + 128)`
separately per sample; a positive integer can impose a smaller cap. This prevents
endless PAD/no-EOS generation. AR decoding skips Lightning sanity checks and
loads no extra model, ASR or audio decoder. Batched mixed-precision generation
can differ from single-sample generation because rounding can change argmax
decisions; use FP32 for stricter comparisons.

### W&B prediction examples

When a `WandbLogger` is configured, regular validation logs
`val/text_only_teacher_forced_examples` from the same existing predictions.
Rows include actual selected input text (raw or normalized), original raw text,
tokenizer, reference/predicted IPA, exact phoneme-token match, first-phoneme
steps, PAD percentages, reference-horizon step count and identifiers. The
`prediction_mode` column explicitly says `teacher_forced`. The `truncated`
column is null because no autoregressive decoding was attempted. Predicted EOS
does not shorten the teacher-forced example's reference horizon.

When AR is explicitly enabled, `val/text_only_decoding_examples` additionally
shows the actual autoregressive outputs and stopping status. Its prediction
mode is `autoregressive`.

`model.text_only_val_wandb_max_examples` defaults to 16 rows globally per table
per validation interval; `0` disables tables. It does not limit scalar metrics.
Rows are gathered from a bounded number of examples per rank, interleaved, and
uploaded once by rank zero at the current trainer step. Speech-only ranks still
participate. Buffers reset each interval. W&B-free runs and sanity checks do not
collect rows. Neither table adds model generation beyond its selected mode.
