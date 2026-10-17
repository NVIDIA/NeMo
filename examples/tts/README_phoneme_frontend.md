# Text-only phoneme frontend for multiturn TTS evaluation

`magpietts_inference.py` accepts an optional text-only `.nemo` frontend when using
`--model_type easy_magpie --easy_magpie_inference_mode multiturn_user_audio`.

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

For every nonempty assistant turn, the frontend receives raw text and a language
tag (`[EN][TEXT_ONLY]`, for example). It runs independent greedy FP32 phoneme
generation with the checkpoint's saved first streaming mode. It does not use
speaker audio, user audio, history, reference IPA, or normalized reference text.
No codec, speaker encoder, or audio predictor is executed by the frontend.
Its phoneme stacking factor must be one. For fixed-delay checkpoints, the
configured expected phoneme delay must match the trained delay. Dynamic-delay
checkpoints ignore that fixed delay, use zero fixed offset, and advance on
predicted PAD/wait tokens. Missing EOS or empty output fails explicitly.

The generated IPA is retokenized using the TTS model's own phoneme tokenizer,
wrapped in its BOS/EOS tokens and installed before each turn's warmup. This is
important because the frontend and TTS model may use different vocabularies.
`gt` selects externally supplied phoneme inputs; those inputs are **frontend
predictions**, not the dataset's ground-truth IPA. TTS still generates its audio
autoregressively, retaining the regular user-audio prefill, speaker context,
conversation cache, CFG, and its own saved phoneme/speech delays.

This is an evaluation cascade with a complete frontend turn generated before
TTS starts that turn. It does not implement simultaneous streaming between two
models. Dynamic waits are supported inside the frontend; the TTS model retains
its own timing. See [dynamic phoneme delay](README_dynamic_phoneme_delay.md).

Each rank loads its own two models. Greedy frontend outputs are cached by raw
text and language across repeats. Output identifiers include the frontend path
identity; turn-level manifests also record its checkpoint SHA256, frontend IPA
and token IDs, the separately retokenized forced TTS IDs, and consumed/total
forced steps. TTS phoneme-head predictions remain separately labeled as
`predicted_phoneme_*`; they are diagnostics, not the forced inputs.

Original normalized text is retained for ASR reference scoring and original IPA
is retained as `reference_phoneme_text`, without becoming generation inputs.
Use immutable model exports and separate result directories when comparing
checkpoints. Disabling the frontend leaves the existing runner path unchanged.

Both models' configured tokenizer files must be readable inside the container.
For the current text-pretraining exports, mount the cluster's `data/models`
directory at `/model_artifacts`; do not override the frontend tokenizer with
the TTS model's 2048-token vocabulary. The dedicated cluster recipe includes
this mount.

Focused tests:

```bash
pytest --cpu -m unit tests/collections/tts/modules/magpietts_inference/test_phoneme_frontend.py
```
