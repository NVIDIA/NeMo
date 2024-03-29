import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from nemo.collections.asr.data.dataclasses import AudioNoiseBatch, AudioNoiseItem
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.core.classes import Exportable, NeuralModule, typecheck


class WavLMAugmentation(object):
    def __init__(
        self,
        prob: float = 0.0,
        noise_ratio: float = 0.0,
        min_r_speech: float = -5.0,
        max_r_speech: float = 5.0,
        min_r_noise: float = -5.0,
        max_r_noise: float = 20.0,
        min_mix_rate: float = 0.0,
        max_mix_rate: float = 1.0,
    ):
        super().__init__()
        self.prob = prob
        self.noise_ratio = noise_ratio
        self.min_r_speech = min_r_speech
        self.max_r_speech = max_r_speech
        self.min_r_noise = min_r_noise
        self.max_r_noise = max_r_noise
        self.min_mix_rate = min_mix_rate
        self.max_mix_rate = max_mix_rate

        if not (0 <= self.prob <= 1):
            raise ValueError(f"prob must be in [0, 1], got: {self.prob}")
        if not (0 <= self.noise_ratio <= 1):
            raise ValueError(f"noise_ratio must be in [0, 1], got: {self.noise_ratio}")
        if not (self.min_r_speech <= self.max_r_speech):
            raise ValueError(
                f"min_r_speech must be no greater than max_r_speech, got: min={self.min_r_speech} and max={self.max_r_speech}"
            )
        if not (self.min_r_noise <= self.max_r_noise):
            raise ValueError(
                f"min_r_noise must be no greater than max_r_noise, got: min={self.min_r_noise} and max={self.max_r_noise}"
            )
        if not (0 <= self.min_mix_rate <= self.max_mix_rate <= 1):
            raise ValueError(
                f"min_mix_rate must be no greater than max_mix_rate, and both must be in [0, 1], got: {self.min_mix_rate} and {self.max_mix_rate}"
            )

    def repeat_noise(self, noise: torch.Tensor, noise_len: int, max_audio_len: int) -> torch.Tensor:
        noise = noise[:noise_len]
        if noise_len < max_audio_len:
            noise = noise.repeat(max_audio_len // noise_len + 1)
        noise = noise[:max_audio_len]
        return noise

    def pad_noise(self, noise: torch.Tensor, max_audio_len: int) -> torch.Tensor:
        noise_len = noise.size(0)
        if noise_len < max_audio_len:
            pad = (0, max_audio_len - noise_len)
            noise = torch.nn.functional.pad(noise, pad)
        else:
            noise = noise[:max_audio_len]
        return noise

    def __call__(self, batch: AudioNoiseBatch) -> AudioNoiseBatch:
        audio_signal = batch.audio
        audio_lengths = batch.audio_len
        batch_size = audio_signal.size(0)
        max_audio_len = audio_signal.size(1)

        noise = batch.noise
        noise_len = batch.noise_len
        noisy_audio = batch.noisy_audio
        noisy_audio_len = batch.noisy_audio_len
        for i in range(batch_size):
            if np.random.rand() > self.prob:
                continue

            # randomly select the length of mixing segment
            if 0 <= self.min_mix_rate < self.max_mix_rate <= 1:
                mix_len = np.random.randint(
                    int(audio_lengths[i] * self.min_mix_rate), int(audio_lengths[i] * self.max_mix_rate)
                )
            else:
                mix_len = max(1, int(audio_lengths[i] * self.min_mix_rate))

            # randomly select position to start the mixing
            mix_start_idx = np.random.randint(audio_lengths[i] - mix_len)

            # randomly select the energy ratio between speech and noise
            if np.random.rand() < self.noise_ratio or batch_size == 1:
                energy_ratio = np.random.uniform(self.min_r_noise, self.max_r_noise)
            else:
                energy_ratio = np.random.uniform(self.min_r_speech, self.max_r_speech)
                j = np.random.choice([x for x in range(batch_size) if x != i])
                noise[i] = audio_signal[j].clone()
                noise_len[i] = audio_lengths[j]

            # repeat noise to match the length of audio mix length if necessary
            if noise_len[i] <= mix_len:
                # repeat noise to match the length of audio mix length
                noise_start_idx = 0
                noise[i] = self.pad_noise(self.repeat_noise(noise[i], noise_len[i], mix_len), max_audio_len)
                noise_len[i] = mix_len
            else:
                # randomly select a segment of noise
                noise_start_idx = np.random.randint(noise_len[i] - mix_len)

            # calculate the scale factor for noise
            audio_energy = torch.sum(audio_signal[i, : audio_lengths[i]] ** 2) / audio_lengths[i]
            noise_energy = torch.sum(noise[i, : noise_len[i]] ** 2) / noise_len[i] if noise_len[i] > 0 else 0
            mix_scale = math.sqrt(audio_energy / (10 ** (energy_ratio / 10) * noise_energy)) if noise_energy > 0 else 0

            # get the residual signal to be added to original audio
            noise_clip = noise[i, noise_start_idx : noise_start_idx + mix_len]
            noise_signal = torch.zeros_like(audio_signal[i])
            noise_signal[mix_start_idx : mix_start_idx + mix_len] = mix_scale * noise_clip

            # add noise to audio
            noisy_audio[i] = audio_signal[i] + noise_signal
            noisy_audio_len[i] = audio_lengths[i]
            noise[i] = noise_signal
            noise_len[i] = audio_lengths[i]

            if noise[i].isnan().any():
                from nemo.utils import logging

                logging.error(f"NaN detected in noise signal")
            if noisy_audio[i].isnan().any():
                from nemo.utils import logging

                logging.error(f"NaN detected in noisy audio signal")

        return AudioNoiseBatch(
            sample_id=batch.sample_id,
            audio=batch.audio,
            audio_len=batch.audio_len,
            noise=noise,
            noise_len=noise_len,
            noisy_audio=noisy_audio,
            noisy_audio_len=noisy_audio_len,
        )
