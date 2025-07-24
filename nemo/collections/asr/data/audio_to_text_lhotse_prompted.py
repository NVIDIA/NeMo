# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch.utils.data
from lhotse import CutSet
from lhotse.cut import MixedCut
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors

from nemo.collections.common.data import apply_prompt_format_fn
from nemo.collections.common.prompts import CanaryPromptFormatter, PromptFormatter
from nemo.collections.common.tokenizers import TokenizerSpec


@dataclass
class PromptedAudioToTextMiniBatch:
    audio: torch.Tensor
    audio_lens: torch.Tensor
    transcript: torch.Tensor
    transcript_lens: torch.Tensor
    prompt: torch.Tensor
    prompt_lens: torch.Tensor
    prompted_transcript: torch.Tensor
    prompted_transcript_lens: torch.Tensor
    cuts: Optional[CutSet] = None

    def get_decoder_inputs_outputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the inputs and outputs of transformer decoder for training.
        The input is ``prompted_transcript`` (minus last token),
        and the output is ``prompted_transcript`` (minus first token).
        """
        return self.prompted_transcript[:, :-1], self.prompted_transcript[:, 1:]


class PromptedAudioToTextLhotseDataset(torch.utils.data.Dataset):
    """
    This dataset is based on :class:`~nemo.collections.asr.data.audio_to_text_lhotse.LhotseSpeechToTextBpeDataset`.
    It is a Lhotse-style dataset that converts a mini-batch of Cuts into tensors.
    The main difference from ``LhotseSpeechToTextBpeDataset`` is that we introduce
    a special prompt format for multitask encoder-decoder models.

    To perform the prompt formatting, we accept a ``prompt_format_fn``.
    It's expected to accept:
    * a ``CutSet`` which it will internally iterate over for utterances, and
    * a ``TokenizerWrapper`` object that will be internally used to tokenize the utterances

    Tokenized utterances will be extended with special prompt tokens according to ``prompt_format_fn`` logic.
    We support cuts with multiple supervision segments -- their tokenized texts will be concatenated before we add the prompt tokens.
    This is useful, for example, in code-switched scenarios where each segment is spoken in a different language.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        prompt: PromptFormatter,
        do_dynamic_chunking: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.padding_value = self.tokenizer.pad_id
        self.prompt = prompt
        self.do_dynamic_chunking = do_dynamic_chunking 

    def __getitem__(self, cuts: CutSet) -> PromptedAudioToTextMiniBatch:
        audio, audio_lens, cuts = self.load_audio(cuts)

        if self.do_dynamic_chunking:            
            new_audio = []
            new_audio_lens = []
            for i in range(audio.shape[0]):
                waveform = audio[i, :audio_lens[i]]
                chunks, chunk_lens = self.chunk_waveform(waveform)
                new_audio.extend(chunks)
                new_audio_lens.extend(chunk_lens)
            audio = torch.stack(new_audio)
            audio_lens = torch.tensor(new_audio_lens, dtype=torch.long)
        # Fast-path: the tokenization and prompt formatting was already done before sampling.
        attrs = ("input_ids", "context_ids", "answer_ids")
        pre_formatted = all(hasattr(c, a) for c in cuts for a in attrs)
        if pre_formatted:
            prompts_with_answers, prompts, answers = zip(*((c.input_ids, c.context_ids, c.answer_ids) for c in cuts))
        else:
            formatted = [apply_prompt_format_fn(cut, self.prompt) for cut in cuts]
            prompts_with_answers = [ex["input_ids"] for ex in formatted]
            prompts = [ex["context_ids"] for ex in formatted]
            answers = [ex["answer_ids"] for ex in formatted]

        transcript, transcript_lens = self._collate_tokens(answers)
        prompts_with_answers, prompts_with_answers_lens = self._collate_tokens(prompts_with_answers)
        prompts, prompt_lens = self._collate_tokens(prompts)

        return PromptedAudioToTextMiniBatch(
            audio=audio,
            audio_lens=audio_lens,
            transcript=transcript,
            transcript_lens=transcript_lens,
            prompt=prompts,
            prompt_lens=prompt_lens,
            prompted_transcript=prompts_with_answers,
            prompted_transcript_lens=prompts_with_answers_lens,
            cuts=_drop_in_memory_data(cuts),
        )

    def _collate_tokens(self, tokens: list[Union[list[int], torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = [torch.as_tensor(t) for t in tokens]
        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=self.padding_value)
        return tokens, token_lens

    def find_optimal_chunk_size(
        self,
        total_len: int,
        min_sec: int = 30,
        max_sec: int = 40,
        sample_rate: int = 16000
    ) -> int:
        best_chunk_size = min_sec * sample_rate
        best_last_chunk_len = 0
        for sec in range(min_sec, max_sec + 1):
            chunk_size = sec * sample_rate
            if chunk_size > total_len:
                continue
            n_chunks = (total_len + chunk_size - 1) // chunk_size  # ceil division
            last_chunk_len = total_len - chunk_size * (n_chunks - 1)
            if last_chunk_len > best_last_chunk_len:
                best_last_chunk_len = last_chunk_len
                best_chunk_size = chunk_size
        return best_chunk_size
            
    def chunk_waveform(self, waveform: torch.Tensor, chunk_size: int = None, device=None) -> tuple[list[torch.Tensor], list[int]]:
        # If chunk_size is None, find the optimal chunk size for this waveform
        total_len = waveform.shape[0]
        if chunk_size is None:
            chunk_size = self.find_optimal_chunk_size(total_len)
        chunks = []
        chunk_lens = []
        start = 0
        while start < total_len:
            end = min(start + chunk_size, total_len)
            chunk = waveform[start:end]
            length = chunk.shape[0]
            if length < chunk_size:
                pad = torch.zeros(chunk_size - length, dtype=chunk.dtype, device=chunk.device if device is None else device)
                chunk = torch.cat([chunk, pad], dim=0)
            chunks.append(chunk)
            chunk_lens.append(length)
            start += chunk_size
        return chunks, chunk_lens

class ProbablyIncorrectLanguageKeyError(RuntimeError):
    pass


def _drop_in_memory_data(
    cuts: CutSet,
    _fields=frozenset(MixedCut.__dataclass_fields__.keys()),
) -> CutSet:
    """Workaround for an edge case in cuts.drop_in_memory_data() on MixedCut with Lhotse<1.29.0"""
    ans = []
    for c in cuts:
        # Not a mixed cut or a mixed cut that wasn't assigned any extra attributes.
        if not isinstance(c, MixedCut) or _fields.issuperset(c.__dict__.keys()):
            ans.append(c.drop_in_memory_data())
        else:
            extra_attrs = {k: v for k, v in c.__dict__.items() if k not in _fields}
            for k in extra_attrs:
                delattr(c, k)
            ans.append(c.drop_in_memory_data())
            for k, v in extra_attrs.items():
                setattr(ans[-1], k, v)
                setattr(c, k, v)
    return CutSet(ans)
