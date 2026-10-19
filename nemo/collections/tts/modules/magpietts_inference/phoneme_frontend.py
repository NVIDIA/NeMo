# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent, text-only phoneme generation for a two-model EasyMagpie cascade."""

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    """Decode one already-tokenized turn, without reference IPA or audio calls.

    Text includes the same BOS/EOS chosen by the loader. The first phoneme step
    counts fixed delay and predicted waits, but excludes the conditioning prefix.
    Incomplete outputs are returned with reached_eos=False for validation metrics.
    With max_steps=None the cap is max(256, 8 * text steps + 128); a positive
    max_steps optionally imposes a smaller ceiling. This function uses the caller's
    precision and never modifies model parameters.
    """
    if (max_steps is not None and (type(max_steps) is not int or max_steps <= 0)) or (
        mode.text_input_mode != 'streaming' or model.phoneme_stacking_factor != 1
    ):
        raise ValueError(
            'Text-only greedy validation requires null or positive max_steps, streaming mode and stacking factor 1'
        )
    if text_tokens.size(0) != 1 or context_tokens.size(0) != 1:
        raise ValueError('Text-only greedy decoding accepts one unpadded turn at a time')
    device = text_tokens.device
    context_lens = torch.tensor([context_tokens.size(1)], device=device, dtype=torch.long)
    text_lens = torch.tensor([text_tokens.size(1)], device=device, dtype=torch.long)
    dynamic = model.cfg.get('dynamic_phoneme_delay', False)
    phoneme_delay = 0 if dynamic else mode.streaming_phonemes_delay
    special_ids = {
        getattr(model.phoneme_tokenizer, 'pad', None),
        model.phoneme_tokenizer.bos_token_id,
        model.phoneme_tokenizer.eos_token_id,
    }
    cas_chunk_size = model.cfg.get('text_only_cas_chunk_size', None)
    context = model.embed_text_tokens(
        context_tokens,
        text_lens=context_lens,
        disable_cas_embedding=model.disable_cas_for_context_text,
        cas_chunk_size=cas_chunk_size,
    )
    if model.task_embedding is not None and mode.mode_idx is not None:
        task = model.task_embedding(torch.tensor([mode.mode_idx], device=device)).unsqueeze(1)
        context = torch.cat([task, context], dim=1)
        context_lens = context_lens + 1
    context_length = context.size(1)
    text_embedded = model.embed_text_tokens(
        text_tokens,
        text_lens=text_lens,
        is_multiturn=model.cfg.get('use_multiturn_dataset', False),
        cas_chunk_size=cas_chunk_size,
    )
    if model.cfg.get('use_multiturn_dataset', False):
        text_embedded[text_tokens == model.pad_id] = 0.0
    delay = context_length + phoneme_delay
    cap = max(256, text_tokens.size(1) * 8 + 128)
    if max_steps is not None:
        cap = min(max_steps, cap)

    def fixed_embeddings(start, length):
        result = context.new_zeros(1, length, context.size(-1))
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
    bos = torch.tensor([[[model.phoneme_tokenizer.bos_token_id]]], device=device)
    prefix[:, delay:] += model.embed_phoneme_tokens(bos)
    output = model.forward(
        inputs_embeds=prefix,
        attention_mask=torch.ones(1, delay + 1, device=device, dtype=torch.bool),
        use_cache=True,
        cache_position=torch.arange(delay + 1, device=device),
        context_prefix_lens=context_lens,
    )
    tokens: List[int] = []
    predicted: List[int] = []
    waits, first = 0, None
    for step in range(cap):
        token = model.phoneme_final_proj(output.last_hidden_state[:, -1:]).argmax(-1)
        token_id = int(token.item())
        predicted.append(token_id)
        if token_id == model.phoneme_tokenizer.eos_token_id:
            return TextOnlyPhonemeTokenOutput(tuple(tokens), step + 1, waits, first, True, tuple(predicted))
        is_wait = dynamic and token_id == model.phoneme_tokenizer.pad
        if first is None and token_id not in special_ids:
            first = phoneme_delay + step + 1
        if is_wait:
            waits += 1
        else:
            tokens.append(token_id)
        if step + 1 == cap:
            break
        position = delay + step + 1
        phoneme_input = model.embed_phoneme_tokens(token.unsqueeze(1))
        if is_wait:
            # Training's multiturn embedding masks PAD to zero. The clock
            # still advances so the next source BPE becomes visible.
            phoneme_input = torch.zeros_like(phoneme_input)
        next_input = fixed_embeddings(position, 1) + phoneme_input
        output = model.forward(
            inputs_embeds=next_input,
            attention_mask=torch.ones(1, position + 1, device=device, dtype=torch.bool),
            use_cache=True,
            past_key_values=output.past_key_values,
            cache_position=torch.tensor([position], device=device),
            context_prefix_lens=context_lens,
        )
    return TextOnlyPhonemeTokenOutput(tuple(tokens), cap, waits, first, False, tuple(predicted))


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


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
