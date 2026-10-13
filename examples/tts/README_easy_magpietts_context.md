# Gated repeated context conditioning in EasyMagpieTTS

`model.repeat_context` defaults to `false`. A missing parameter also uses `false`,
preserving the existing prefix-only training and inference behavior.

To enable the context channel for a new training run, use:

```bash
python examples/tts/easy_magpietts.py model.repeat_context=true <other training overrides>
```

The `easy_magpietts_lhotse.yaml` and `easy_magpietts_lhotse_multiturn.yaml` configurations
both expose this parameter. It is a model setting, saved with the checkpoint; use
the same setting during inference. Train or fine-tune with it enabled before
evaluating the changed conditioning behavior on an existing checkpoint.

With repetition enabled, the model tiles each item's valid **prepared context
sequence** across the decoder timeline, scales it through a learned gate, and sums it with the input channels:

```text
decoder_input[t] = text_channel[t] + phoneme_channel[t] + audio_channel[t]
                   + user_audio_channel[t] + gain * prepared_context[t % context_length]
gain = softplus(context_gate_logits)
```

The gain is a learned vector with one value per embedding dimension (1,536 parameters
for an embedding dimension of 1,536). It is initialized to **1** using inverse
softplus, so conditioning contributes immediately when training from scratch.
The gain stays nonnegative and has no upper cap: training can reduce or amplify
each conditioning feature. It is shared across sequence positions and requests;
it is not an input-dependent gate. Enabled repetition always uses this gate.

The prepared context contains the existing task embedding, when present, followed
by the speaker reference and context text. This repeats the sequence rather than
pooling it into one vector. The first occurrence remains the original context
prefix and is gated and added only once. Each batch item repeats its own valid context;
batch padding and positions beyond its training sequence length contribute zero.
For text-only batches, the context contains only the task embedding and context
text, and no audio conditioning is introduced.

The repeated channel reuses embeddings after the existing speaker encoder and
training-only zero-shot blocker (`train_shuffle_context_embedding_prob`). It does
not provide a new path around either. When classifier-free guidance drops
conditioning, the repeated channel contains only the unconditional context token.
The unconditional inference branch likewise repeats that token without the
speaker reference or context text. The same gate scales the unconditional token.

Streaming initialization gates the context prefix. Prefill and cached
decoding then use absolute decoder positions to continue the repetition, including
across conversation turns. Splitting a prefill into chunks does not restart the
context sequence. The reference and context text are encoded once per request,
not again at every decoding step. The gate is applied exactly once during training,
initialization, prefill and each decoding step. It does not impose a context window
or reset the decoder cache. With `repeat_context=false` or a missing parameter,
no gate weights are created, and the original prefix-only behavior is preserved.
