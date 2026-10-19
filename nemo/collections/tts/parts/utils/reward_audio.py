# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Waveform preparation shared by online preference-optimization rewards."""

from typing import List, Sequence

import torch
from torchaudio.functional import resample


def split_audio_batch(audio: torch.Tensor, lengths: torch.Tensor) -> List[torch.Tensor]:
    """Trim padded mono audio to its true lengths and detach it from the training graph."""
    if audio.ndim != 2 or lengths.ndim != 1 or audio.shape[0] != lengths.numel():
        raise ValueError('Expected audio [B, T] and one length per waveform.')
    sizes = lengths.detach().cpu().tolist()
    if any(int(size) != size or size <= 0 or size > audio.shape[1] for size in sizes):
        raise ValueError('Audio lengths must be positive integers within the padded width.')
    audio = audio.detach().float()
    return [row[: int(size)] for row, size in zip(audio, sizes)]


def resample_audio(audio: Sequence[torch.Tensor], sample_rate: int, target_rate: int) -> List[torch.Tensor]:
    """Resample detached mono waveforms on their existing device without padding them."""
    if sample_rate <= 0 or target_rate <= 0:
        raise ValueError('Sample rates must be positive.')
    result = []
    for waveform in audio:
        if waveform.ndim != 1 or waveform.numel() == 0:
            raise ValueError('Expected nonempty mono waveforms.')
        waveform = waveform.detach().float()
        result.append(resample(waveform, sample_rate, target_rate) if sample_rate != target_rate else waveform)
    return result
