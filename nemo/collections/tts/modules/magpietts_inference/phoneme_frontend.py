# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent, text-only phoneme generation for a two-model EasyMagpie cascade."""

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from omegaconf import open_dict

from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import _strip_timestamps
from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel


@dataclass(frozen=True)
class PhonemeFrontendOutput:
    """IPA and frontend token IDs, excluding BOS and EOS."""

    ipa: str
    token_ids: Tuple[int, ...]
    decoder_steps: int
    wait_tokens: int = 0
    first_phoneme_step: int | None = None


@dataclass(frozen=True)
class TextOnlyPhonemeTokenOutput:
    """Cached greedy decoding; first_phoneme_step is one-based, excluding context."""

    token_ids: Tuple[int, ...]
    decoder_steps: int
    wait_tokens: int
    first_phoneme_step: int | None
    reached_eos: bool
    predicted_tokens: Tuple[int, ...]


@torch.inference_mode()
def generate_text_only_phoneme_tokens(model, text_tokens, context_tokens, mode, max_steps):
    """Decode one unpadded turn using the same batched cached validation path."""
    if text_tokens.size(0) != 1 or context_tokens.size(0) != 1:
        raise ValueError('Text-only greedy decoding accepts one unpadded turn at a time')
    return generate_batched_text_only_phoneme_tokens(
        model,
        text_tokens,
        text_tokens.new_tensor([text_tokens.size(1)]),
        context_tokens,
        context_tokens.new_tensor([context_tokens.size(1)]),
        mode,
        max_steps,
    )[0]


@torch.inference_mode()
def generate_batched_text_only_phoneme_tokens(
    model, text_tokens, text_lens, context_tokens, context_lens, mode, max_steps
):
    """Decode padded source batches with independent EOS and safety-cap stopping.

    Returns outputs in input order. Equal-length conditioning prefixes share one
    cache and autoregressive loop; different prefix lengths use separate cohorts
    to avoid introducing padding steps into the Mamba state. Source text lengths
    may differ within a cohort. Ground-truth phonemes are never decoder inputs.
    A null max_steps uses max(256, 8 * source steps + 128) separately per sample.
    """
    if (max_steps is not None and (type(max_steps) is not int or max_steps <= 0)) or (
        mode.text_input_mode != 'streaming' or model.phoneme_stacking_factor != 1
    ):
        raise ValueError(
            'Text-only greedy validation requires null or positive max_steps, streaming mode and stacking factor 1'
        )
    batch_size = text_tokens.size(0)
    if context_tokens.size(0) != batch_size or text_lens.shape != (batch_size,) or context_lens.shape != (batch_size,):
        raise ValueError('Source text, contexts and lengths must have the same batch size')
    source_lengths, prefix_lengths = text_lens.tolist(), context_lens.tolist()
    if any(not 0 < length <= text_tokens.size(1) for length in source_lengths) or any(
        not 0 <= length <= context_tokens.size(1) for length in prefix_lengths
    ):
        raise ValueError('Source lengths must describe the unpadded text and context tokens')
    cohorts = {}
    for index, length in enumerate(prefix_lengths):
        cohorts.setdefault(length, []).append(index)
    results = [None] * batch_size
    for prefix_length, indices in cohorts.items():
        index = text_tokens.new_tensor(indices)
        lengths = text_lens.index_select(0, index)
        source_width = max(source_lengths[i] for i in indices)
        decoded = _decode_text_only_phoneme_cohort(
            model,
            text_tokens.index_select(0, index)[:, :source_width],
            lengths,
            context_tokens.index_select(0, index)[:, :prefix_length],
            context_lens.index_select(0, index),
            mode,
            max_steps,
        )
        for i, output in zip(indices, decoded):
            results[i] = output
    return results


class TextOnlyPhonemeFrontend:
    """Greedy FP32 decoding with fixed delay or alignment-trained waits.

    Each call starts a fresh text-only cache. Speaker audio, conversation history,
    normalized reference text and reference IPA are never frontend inputs.
    """

    def __init__(self, checkpoint: str, device, codecmodel_path=None, expected_delay: int = 8, max_steps: int = 4096):
        if max_steps <= 0:
            raise ValueError("phoneme_frontend_max_steps must be positive")
        self.checkpoint = os.path.abspath(checkpoint)
        self.max_steps = max_steps
        target_device = torch.device(device)
        rng_devices = [target_device] if target_device.type == 'cuda' else []
        # Loading another model must not advance the TTS sampling RNG.
        with torch.random.fork_rng(devices=rng_devices):
            self.model = self._load_model(codecmodel_path, target_device)
        if self.model.phoneme_tokenizer is None or self.model.phoneme_stacking_factor != 1:
            raise ValueError("The text-only frontend requires a phoneme tokenizer with stacking factor 1")
        self.mode = self.model.training_modes[0]
        self.dynamic_phoneme_delay = self.model.cfg.get('dynamic_phoneme_delay', False)
        if self.dynamic_phoneme_delay and getattr(self.model.phoneme_tokenizer, 'pad', None) is None:
            raise ValueError('Dynamic phoneme delay requires a phoneme PAD token for waits')
        if self.mode.text_input_mode != 'streaming' or (
            not self.dynamic_phoneme_delay and self.mode.streaming_phonemes_delay != expected_delay
        ):
            raise ValueError(
                f"Expected a streaming frontend with phoneme delay {expected_delay}; got {self.mode.name}. "
                "Select the correct checkpoint; do not override its trained delay."
            )
        self.phoneme_delay = 0 if self.dynamic_phoneme_delay else self.mode.streaming_phonemes_delay
        self.metadata = dict(
            checkpoint=self.checkpoint,
            checkpoint_sha256=_file_sha256(self.checkpoint),
            phoneme_delay=self.phoneme_delay,
            max_steps=self.max_steps,
            decoding='greedy FP32, TF32 disabled, independent raw-text turns',
        )
        self._cache: Dict[Tuple[str, str], PhonemeFrontendOutput] = {}
        if self.dynamic_phoneme_delay:
            self.metadata['dynamic_phoneme_delay'] = True
            self.metadata['configured_phoneme_delay'] = self.mode.streaming_phonemes_delay

    def generate(self, text: str, language: str) -> PhonemeFrontendOutput:
        """Generate IPA from raw text only; never fall back to reference phonemes."""
        key = (text, language)
        if key not in self._cache:
            old_matmul = torch.backends.cuda.matmul.allow_tf32
            old_cudnn = torch.backends.cudnn.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                with torch.inference_mode():
                    self._cache[key] = self._generate(text, language)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = old_matmul
                torch.backends.cudnn.allow_tf32 = old_cudnn
        return self._cache[key]

    def _generate(self, text: str, language: str) -> PhonemeFrontendOutput:
        if not isinstance(text, str) or not text.strip() or not language:
            raise ValueError("The phoneme frontend requires nonempty raw text and a language")
        model, device = self.model, self.model.device
        context_ids = model.tokenizer.encode(
            f'[{language.upper()}][TEXT_ONLY]', tokenizer_name=model.text_conditioning_tokenizer_name
        )
        text_ids = model.tokenizer.encode(
            _strip_timestamps(text), tokenizer_name=model.text_conditioning_tokenizer_name
        ) + [model.eos_id]
        if self.dynamic_phoneme_delay and model.cfg.get('add_text_bos', False):
            text_ids.insert(0, model.bos_id)
        result = generate_text_only_phoneme_tokens(
            model,
            torch.tensor([text_ids], device=device, dtype=torch.long),
            torch.tensor([context_ids], device=device, dtype=torch.long),
            self.mode,
            self.max_steps,
        )
        if not result.reached_eos:
            raise RuntimeError(
                f"The phoneme frontend did not reach EOS within {result.decoder_steps} steps; "
                "no reference fallback is allowed"
            )
        ipa = model.phoneme_tokenizer.decode(list(result.token_ids)).strip()
        if not ipa:
            raise RuntimeError("The phoneme frontend generated empty IPA; no reference fallback is allowed")
        return PhonemeFrontendOutput(
            ipa, result.token_ids, result.decoder_steps, result.wait_tokens, result.first_phoneme_step
        )

    def _load_model(self, codecmodel_path, device):
        cfg = EasyMagpieTTSInferenceModel.restore_from(self.checkpoint, return_config=True)
        with open_dict(cfg):
            cfg.target = 'nemo.collections.tts.models.easy_magpietts_inference.EasyMagpieTTSInferenceModel'
            cfg.train_ds = None
            cfg.validation_ds = None
            cfg.test_ds = None
            cfg.run_val_inference = False
            cfg.use_utmos = False
            if codecmodel_path is not None:
                cfg.codecmodel_path = codecmodel_path
        return (
            EasyMagpieTTSInferenceModel.restore_from(
                self.checkpoint, override_config_path=cfg, map_location='cpu', strict=True
            )
            .eval()
            .requires_grad_(False)
            .float()
            .to(device)
        )


def _decode_text_only_phoneme_cohort(model, text_tokens, text_lens, context_tokens, context_lens, mode, max_steps):
    device, batch_size = text_tokens.device, text_tokens.size(0)
    dynamic = model.cfg.get('dynamic_phoneme_delay', False)
    phoneme_delay = 0 if dynamic else mode.streaming_phonemes_delay
    tokenizer = model.phoneme_tokenizer
    pad = getattr(tokenizer, 'pad', None)
    cas_chunk_size = model.cfg.get('text_only_cas_chunk_size', None)
    context = model.embed_text_tokens(
        context_tokens,
        text_lens=context_lens,
        disable_cas_embedding=model.disable_cas_for_context_text,
        cas_chunk_size=cas_chunk_size,
    )
    if model.task_embedding is not None and mode.mode_idx is not None:
        task = model.task_embedding(torch.full((batch_size,), mode.mode_idx, device=device, dtype=torch.long))
        context = torch.cat([task.unsqueeze(1), context], dim=1)
        context_lens = context_lens + 1
    context_length = context.size(1)
    text_embedded = model.embed_text_tokens(
        text_tokens,
        text_lens=text_lens,
        is_multiturn=model.cfg.get('use_multiturn_dataset', False),
        cas_chunk_size=cas_chunk_size,
    )
    # Zero batch padding even when it contains non-PAD IDs. Each source stops at
    # its own valid length, exactly as in independent unpadded decoding.
    invalid = torch.arange(text_tokens.size(1), device=device)[None, :] >= text_lens[:, None]
    if model.cfg.get('use_multiturn_dataset', False):
        invalid = invalid | text_tokens.eq(model.pad_id)
    text_embedded = text_embedded.masked_fill(invalid[:, :, None], 0)
    delay = context_length + phoneme_delay
    caps = (8 * text_lens + 128).clamp_min(256)
    if max_steps is not None:
        caps = caps.clamp_max(max_steps)
    cap = int(caps.max())

    def fixed_embeddings(start, length):
        result = context.new_zeros(batch_size, length, context.size(-1))
        if model.repeat_context:
            positions = torch.arange(start, start + length, device=device)
            result += model._repeat_gated_context_embeddings(context, context_lens, positions)
        elif start < context_length:
            end = min(start + length, context_length)
            result[:, : end - start] += context[:, start:end]
        begin = max(start, context_length)
        end = min(start + length, context_length + text_embedded.size(1))
        if begin < end:
            result[:, begin - start : end - start] += text_embedded[:, begin - context_length : end - context_length]
        return result

    prefix = fixed_embeddings(0, delay + 1)
    bos = torch.full((batch_size, 1, 1), tokenizer.bos_token_id, device=device, dtype=torch.long)
    prefix[:, delay:] += model.embed_phoneme_tokens(bos)
    output = model.forward(
        inputs_embeds=prefix,
        attention_mask=torch.ones(batch_size, delay + 1, device=device, dtype=torch.bool),
        use_cache=True,
        cache_position=torch.arange(delay + 1, device=device),
        context_prefix_lens=context_lens,
    )
    predicted = text_tokens.new_full((batch_size, cap), tokenizer.eos_token_id)
    done = torch.zeros(batch_size, device=device, dtype=torch.bool)
    steps = torch.zeros_like(text_lens)
    for step in range(cap):
        token = model.phoneme_final_proj(output.last_hidden_state[:, -1:]).argmax(-1).squeeze(1)
        predicted[:, step] = token
        steps += (~done).long()
        done = done | token.eq(tokenizer.eos_token_id) | caps.le(step + 1)
        if done.all():
            break
        # Finished rows keep their cache slot until all rows end. Their later
        # predictions are ignored; no cache compaction or cross-row state reuse.
        token = token.masked_fill(done, tokenizer.eos_token_id)
        phoneme_input = model.embed_phoneme_tokens(token[:, None, None])
        zero_input = done | (token.eq(pad) if dynamic else torch.zeros_like(done))
        phoneme_input = phoneme_input.masked_fill(zero_input[:, None, None], 0)
        position = delay + step + 1
        output = model.forward(
            inputs_embeds=fixed_embeddings(position, 1) + phoneme_input,
            attention_mask=torch.ones(batch_size, position + 1, device=device, dtype=torch.bool),
            use_cache=True,
            past_key_values=output.past_key_values,
            cache_position=torch.tensor([position], device=device),
            context_prefix_lens=context_lens,
        )
    # One transfer per cohort, not a host synchronization for every sample at
    # every generated step. Do not retain the full GPU logit history.
    predictions = predicted[:, : step + 1].tolist()
    lengths = steps.tolist()
    results = []
    special_ids = {pad, tokenizer.bos_token_id, tokenizer.eos_token_id}
    for row, length in zip(predictions, lengths):
        row = row[:length]
        eos = bool(row and row[-1] == tokenizer.eos_token_id)
        tokens = row[:-1] if eos else row
        waits = tokens.count(pad) if dynamic else 0
        content = tuple(token for token in tokens if not dynamic or token != pad)
        first = next((phoneme_delay + i + 1 for i, token in enumerate(row) if token not in special_ids), None)
        results.append(TextOnlyPhonemeTokenOutput(content, length, waits, first, eos, tuple(row)))
    return results


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
