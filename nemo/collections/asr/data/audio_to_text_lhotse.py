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

from typing import Dict, Optional, Tuple

import torch
import torch.utils.data
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors

from nemo.collections.common.tokenizers.aggregate_tokenizer import TokenizerWrapper
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType


class LhotseSpeechToTextBpeDataset(torch.utils.data.Dataset):
    """
    This dataset is based on BPE datasets from audio_to_text.py.
    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.
    Chunking:
    If `enable_chunking` is True, each audio sample is split into optimally sized chunks
    (see `_find_optimal_chunk_size` and `_chunk_waveform`). This is useful for long audio inputs,
    allowing the model to process them in manageable segments. Note that when chunking is enabled,
    the same transcript tokens are replicated for each audio chunk.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal()),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
            'transcripts': NeuralType(('B', 'T'), LabelsType()),
            'transcript_length': NeuralType(tuple('B'), LengthsType()),
            'sample_id': NeuralType(tuple('B'), LengthsType(), optional=True),
        }

    def __init__(self, tokenizer: TokenizerSpec, return_cuts: bool = False, enable_chunking: bool = True):
        super().__init__()
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.return_cuts = return_cuts
        self.enable_chunking = enable_chunking

    def __getitem__(self, cuts) -> Tuple[torch.Tensor, ...]:
        audio, audio_lens, cuts = self.load_audio(cuts)
        # Handle chunking if enabled
        original_tokens = None
        if self.enable_chunking:
            # Store original tokens before chunking audio
            original_tokens = [
                torch.cat(
                    [
                        torch.as_tensor(s.tokens if hasattr(s, "tokens") else self.tokenizer(s.text or "", s.language))
                        for s in c.supervisions
                    ],
                    dim=0,
                )
                for c in cuts
            ]

            # Chunk the audio
            new_audio = []
            new_audio_lens = []
            new_tokens = []

            for i in range(audio.shape[0]):
                waveform = audio[i, : audio_lens[i]]
                # Split the waveform into chunks and get their lengths
                chunks, chunk_lens = self._chunk_waveform(waveform)
                new_audio.extend(chunks)
                new_audio_lens.extend(chunk_lens)
                # Replicate the same tokens for each chunk
                new_tokens.extend([original_tokens[i]] * len(chunks))

            # Stack all chunks into a batch
            audio = torch.stack(new_audio)
            audio_lens = torch.tensor(new_audio_lens, dtype=torch.long)
            tokens = new_tokens
        else:
            # Original tokenization logic when chunking is disabled
            tokens = [
                torch.cat(
                    [
                        torch.as_tensor(s.tokens if hasattr(s, "tokens") else self.tokenizer(s.text or "", s.language))
                        for s in c.supervisions
                    ],
                    dim=0,
                )
                for c in cuts
            ]

        token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)
        tokens = collate_vectors(tokens, padding_value=0)

        if self.return_cuts:
            return audio, audio_lens, tokens, token_lens, cuts.drop_in_memory_data()
        return audio, audio_lens, tokens, token_lens

    def _find_optimal_chunk_size(
        self, total_len: int, min_sec: int = 30, max_sec: int = 40, sample_rate: int = 16000, overlap_sec: float = 1.0
    ) -> int:
        """
        Find the optimal chunk size for audio processing that minimizes paddings to the last chunk.

        Args:
            total_len (int): Total length of the audio waveform in samples
            min_sec (int, optional): Minimum chunk size in seconds. Defaults to 30.
            max_sec (int, optional): Maximum chunk size in seconds. Defaults to 40.
            sample_rate (int, optional): Audio sample rate in Hz. Defaults to 16000.
            overlap_sec (float, optional): Overlap duration between consecutive chunks in seconds.
                                         Defaults to 1.0.

        Returns:
            int: Optimal chunk size in samples that maximizes the last chunk length
        """
        best_chunk_size = min_sec * sample_rate
        best_last_chunk_len = 0
        if total_len < max_sec * sample_rate:
            return total_len
        # Try each possible chunk duration in the range
        for sec in range(min_sec, max_sec + 1):
            chunk_size = sec * sample_rate
            overlap_size = int(overlap_sec * sample_rate)
            step_size = chunk_size - overlap_size

            if step_size <= 0:  # Invalid overlap
                continue
            if chunk_size > total_len:
                continue

            # Calculate how many chunks we'd need and the last chunk's length
            n_chunks = (total_len + step_size - 1) // step_size
            last_chunk_len = total_len - step_size * (n_chunks - 1)

            if last_chunk_len > best_last_chunk_len:
                best_last_chunk_len = last_chunk_len
                best_chunk_size = chunk_size

        return best_chunk_size

    def _chunk_waveform(
        self, waveform: torch.Tensor, chunk_size: int = None, overlap_sec: float = 1.0, sample_rate: int = 16000
    ) -> Tuple[list[torch.Tensor], list[int]]:
        """
        Split a waveform tensor into overlapping chunks.

        Args:
            waveform (torch.Tensor): Input audio waveform tensor of shape (time_samples,)
            chunk_size (int, optional): Size of each chunk in samples. If None, automatically
                                       determines optimal chunk size using _find_optimal_chunk_size().
                                       Defaults to None.
            sample_rate (int, optional): Audio sample rate in Hz. Defaults to 16000.
            overlap_sec (float, optional): Overlap duration between consecutive chunks in seconds.
                                          Used to calculate step size. Defaults to 1.0.

        Returns:
            Tuple[list[torch.Tensor], list[int]]: A tuple containing:
                - List of chunk tensors, each of shape (chunk_size,)
                - List of original lengths for each chunk before padding (useful for masking
                  padded regions during processing.
        """
        # If chunk_size is None, find the optimal chunk size for this waveform
        total_len = waveform.shape[0]
        if chunk_size is None:
            chunk_size = self._find_optimal_chunk_size(total_len, overlap_sec=overlap_sec)
        if chunk_size >= total_len:
            return [waveform], [total_len]
        overlap_size = int(overlap_sec * sample_rate)
        step_size = chunk_size - overlap_size
        chunks = []
        chunk_lens = []
        start = 0
        while start + overlap_size < total_len:
            end = min(start + chunk_size, total_len)
            chunk = waveform[start:end]
            length = chunk.shape[0]
            if length < chunk_size:
                pad = torch.zeros(chunk_size - length, dtype=chunk.dtype, device=chunk.device)
                chunk = torch.cat([chunk, pad], dim=0)
            chunks.append(chunk)
            chunk_lens.append(length)
            start += step_size

        return chunks, chunk_lens
