# Pinned-prefix sliding-window attention in Nemotron-H

`model.nemotron_h_config.rswa_window` defaults to `null`. A missing setting also
preserves the original full-attention path, including the selected SDPA or
FlashAttention2 backend. No weights are added or renamed.

Enable R-SWA when training a Nemotron-H decoder with, for example:

```bash
python examples/tts/easy_magpietts.py --config-name=easy_magpietts_lhotse \
  model.decoder_type=nemotron_h model.nemotron_h_config.rswa_window=3000 \
  model.repeat_context=false <other training overrides>
```

Both Lhotse single-turn and multiturn YAML configurations expose the setting.
`3000` is an example window, not a profiled safe training-duration cap. At 25
codec frames/second and frame stacking of 2, it corresponds to approximately four
minutes of speech. The unit is **decoder steps**, including the current step;
conditioning-prefix steps are retained in addition to the rolling window.

For query position `q`, key position `k`, window `W`, and the item's fixed prepared
conditioning-prefix length `P`, attention is allowed exactly when:

```text
query and key are valid, k <= q, and (k < P or q - k < W)
```

This is the visibility rule used by upstream vLLM R-SWA. The prefix remains
causal while it is being processed: a prefix query cannot attend to future
prefix tokens. Each batch item has its own boundary, taken from the existing
prepared task/speaker-reference/context-text sequence. Text-only training pins
its task/context-text prefix and introduces no audio conditioning. Batch padding
is excluded. The existing zero-shot blocker and classifier-free conditioning
dropout still apply before these embeddings reach the decoder.

All Nemotron-H `*` attention layers use this mask; Mamba, MoE, CAS and the local
audio Transformer are unchanged. R-SWA and gated `repeat_context` are independent
options and can be combined; keep repetition disabled for an attention-only
ablation. The window setting is saved in the model configuration, so use the
same value when restoring a checkpoint for NeMo inference.

CUDA uses compiled PyTorch FlexAttention with a block-sparse mask built once per
decoder forward and shared by the attention layers, retaining grouped-query
attention without expanding KV heads. It bypasses FlashAttention2 when enabled.
The tested runtime is PyTorch 2.12 on A100; older builds require compatible
FlexAttention support. `attention_dropout` must be zero. First use incurs kernel
compilation; varying sequence lengths use dynamic compilation. CPU has a dense
SDPA reference path intended for small correctness tests, not long training.

NeMo streaming initialization, prefill and cached decoding pass the **original
conditioning boundary**, including for both classifier-free guidance branches.
Appending a turn's text does not enlarge the pinned prefix. Absolute positions
continue across turns. Direct decoder callers must provide `context_prefix_lens`
on every enabled forward, and a padding mask, if present, must cover all cached
and current keys. Activation checkpointing uses the same visibility mask during
recomputation.

This change implements training and NeMo inference visibility. It does **not**
compact or evict the existing NeMo KV cache: cached decoding retains old keys in
memory and masks them out. Bounded inference-cache memory requires separate cache
eviction or an R-SWA-capable serving runtime. The project's vLLM-Omni adapter needs
its own integration and must pin the initial conditioning boundary rather than
every later text prompt extension. No vLLM-Omni files are changed here.

In full-text input mode, target text lies after the conditioning prefix and can
leave the window before all of it is spoken. Choose a window that covers the
required text/audio alignment or feed text in short streaming/continuation turns.
This option does not automatically protect pending target text, reset Mamba
state, or guarantee a quality improvement or a higher safe training-duration cap.

References:

- [R-SWA implementation in vLLM](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/layers/attention/rswa_attention.py)
- [Unlimited OCR Works](https://arxiv.org/abs/2606.23050)
- [PyTorch FlexAttention](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html)
