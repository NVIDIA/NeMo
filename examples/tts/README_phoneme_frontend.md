# Text-only phoneme frontend for TTS evaluation

`magpietts_inference.py` accepts an optional text-only `.nemo` frontend when using
`--model_type easy_magpie`, with either `single_turn` (the default) or
`multiturn_user_audio`. Both modes require `--phoneme_input_type gt`.

```bash
torchrun --standalone --nproc_per_node=8 examples/tts/magpietts_inference.py \
  --model_type easy_magpie \
  --easy_magpie_inference_mode multiturn_user_audio \
  --nemo_files /absolute/path/to/tts.nemo \
  --codecmodel_path /absolute/path/to/codec.nemo \
  --datasets_json_path /absolute/path/to/evalsets.json \
  --out_dir /absolute/path/to/cascade_results \
  --batch_size 1 --max_eval_turns 3 --num_repeats 3 \
  --phoneme_frontend_nemo_file /absolute/path/to/text_only_frontend.nemo \
  --phoneme_frontend_expected_delay 8 \
  --phoneme_input_type gt --phoneme_sampling_method argmax \
  --use_local_transformer --use_cfg --cfg_scale 2.5 \
  --run_evaluation
```

For every single-turn sample or nonempty assistant turn, the frontend receives raw text and a language
tag (`[EN][TEXT_ONLY]`, for example). It runs independent greedy FP32 phoneme
generation with the checkpoint's saved first streaming mode. It does not use
speaker audio, user audio, history, reference IPA, or normalized reference text.
No codec, speaker encoder, or audio predictor is executed by the frontend.
Its phoneme stacking factor must be one. For fixed-delay checkpoints, the
configured expected phoneme delay must match the trained delay. Dynamic-delay
checkpoints ignore that fixed delay, use zero fixed offset, and advance on
predicted PAD/wait tokens. Decoding retains the adaptive safety budget
`max(256, 8 * source tokens + 128)`, capped by `--phoneme_frontend_max_steps`.
A missing EOS may indicate a model output loop; increasing the limit does not
necessarily recover the turn.

Missing EOS fails by default. For complete dataset scoring, select
`--phoneme_frontend_on_max_steps use_partial`: nonempty capped predictions are
passed to TTS and scored, with `frontend_reached_eos=false` and
`frontend_stop_reason=max_steps` in each output record. They are never replaced
with reference IPA or dropped from scoring. Empty IPA still fails explicitly.

When both phoneme tokenizers have identical token-to-ID mappings and special
IDs, frontend token IDs are passed directly to TTS: PAD/wait tokens are removed,
and one TTS BOS/EOS pair wraps the remaining IDs. This preserves the frontend's
exact token sequence. Different vocabularies use decoded IPA retokenized with
the TTS tokenizer. Saved records label this `phoneme_token_transfer` as
`direct_ids` or `retokenized_ipa`. Inputs are installed before turn warmup.
`gt` selects externally supplied phoneme inputs; those inputs are **frontend
predictions**, not the dataset's ground-truth IPA. TTS still generates its audio
autoregressively, retaining the regular user-audio prefill, speaker context,
conversation cache, CFG, and its own saved phoneme/speech delays.

This is an evaluation cascade with a complete frontend turn generated before
TTS starts that turn. It does not implement simultaneous streaming between two
models. Dynamic waits are supported inside the frontend; the TTS model retains
its own timing. See [dynamic phoneme delay](README_dynamic_phoneme_delay.md).

The frontend creates fresh attention and Mamba caches for each independent
turn; the TTS conversation cache continues normally.

Single-turn evaluation supports batched speech generation and writes
`phoneme_frontend_manifest.jsonl` alongside each repeat's audio, recording raw
text, generated IPA and both tokenizations.

Each rank loads its own two models. Greedy frontend outputs are cached by raw
text and language across repeats. Output identifiers include the frontend path
identity; turn-level manifests also record its checkpoint SHA256, frontend IPA
and token IDs, the forced TTS IDs, and consumed/total
forced steps. TTS phoneme-head predictions remain separately labeled as
`predicted_phoneme_*`; they are diagnostics, not the forced inputs.

Original normalized text is retained for ASR reference scoring and original IPA
is retained as `reference_phoneme_text`, without becoming generation inputs.
Use immutable model exports and separate result directories when comparing
checkpoints. Disabling the frontend leaves the existing runner path unchanged.

Both models' configured tokenizer files must be readable inside the container.
For the current text-pretraining exports, mount the cluster's `data/models`
directory at `/model_artifacts`; do not override the frontend tokenizer with
a different TTS phoneme vocabulary. The dedicated cluster recipe includes
this mount.

Focused tests:

```bash
pytest --cpu -m unit tests/collections/tts/modules/magpietts_inference/test_phoneme_frontend.py
```

The cluster launcher `run_all_evals_for_checkpoint_new_runner_with_retries_text_frontend.sh`
uses the full frontend experiment name in the results directory, requests CPU
cores and host RAM in proportion to GPU count, and selects `use_partial`.
Frontend evaluation defaults to one repeat to reduce wall time under the
four-hour GPU partition limit. `NUM_REPEATS` overrides that count;
without a frontend the existing three-repeat default is preserved.


Single-turn normalization evaluation datasets may set `"load_normalized_text_percent": 0.0`
in their dataset JSON entry to feed the manifest's raw `text` to the speech model.
Keep `normalized_text` as the evaluation reference and `original_text` as the raw text
for an optional phoneme frontend. Existing dataset configurations retain their default
of `1.0` (use `normalized_text` when available). This option applies to the single-turn
EasyMagpie runner; it does not change multiturn inference.

## Memory-bounded, multi-GPU single-turn evaluation

Single-turn EasyMagpie evaluation supports independent batched generation on each
GPU with `torchrun --standalone --nproc_per_node=8`. Keep the usual arguments and
select `--easy_magpie_inference_mode single_turn`. `--batch_size` is the batch
size **per GPU**, so eight workers with `--batch_size 12` can generate up to 96
samples concurrently. Each worker restores the model on its own CUDA device.
This also works with the optional phoneme frontend; each worker loads both models
and the frontend starts fresh conversation caches for each sample.

Use `--codec_decode_batch_size 1` to decode generated codec tokens one sample at a
time. Sub-batches are cropped to their own maximum valid length, avoiding waveform
decoder memory growth from padding to another sample's long generation. The
default preserves the checkpoint's setting; `0` requests full-batch decoding.
This option changes waveform decoding memory use, not autoregressive generation
batch size. Small floating-point differences from full-batch codec decoding can
occur. Generation that reaches `--max_decoder_steps` keeps its partial audio and
logs a warning plus `samples_without_audio_eos` in the timing metrics.

Workers write separate rank directories and completion manifests. Rank zero
validates the shards, merges outputs using original manifest indices, and runs the
existing evaluator against the original references. ASR scoring remains on rank
zero. A single-process run keeps the existing output layout and naming.
