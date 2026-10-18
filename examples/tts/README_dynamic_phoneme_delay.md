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
source offsets through timestamp removal/whitespace collapse, and gates each
changed/entity IPA span until the whole source expression and its closing
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

Valid expression groups in `partial` samples remain usable. A bounded unresolved
region can be recovered as a complete many-to-many group when its normalized IPA
boundaries are known. Missing, stale, malformed, or otherwise unusable metadata
falls back conservatively to waiting for the selected turn's full text EOS
before producing that turn's first IPA token; it never guesses missing offsets.

Training logs aggregate across all ranks, including mixed speech/text batches:

- `train/dynamic_phoneme_wait_tokens`: inserted waits per global microbatch.
- `train/dynamic_phoneme_wait_percent`: waits / valid phoneme-channel tokens,
  including BOS/EOS in the denominator.
- `train/dynamic_phoneme_fallback_percent`: turns using full-text fallback / text
  turns. A high rate means the intended aligned streaming supervision is absent.

`TextOnlyPhonemeFrontend` reads the checkpoint's saved flag. PAD predictions
advance the text/decoder clock with zero phoneme input and are omitted from the
returned IPA/token IDs. No reference alignment is needed at inference. Its
`wait_tokens` output reports generated waits. `--phoneme_frontend_expected_delay`
guards the trained delay for fixed-delay checkpoints; dynamic checkpoints ignore
this argument and always use zero fixed phoneme delay. Frontend metadata reports
the effective `phoneme_delay: 0` and the ignored `configured_phoneme_delay`.

This feature does not shift audio targets or implement dynamic-delay joint speech
generation. In the two-model cascade, the frontend finishes IPA generation before
the existing TTS model consumes the resulting phonemes.
