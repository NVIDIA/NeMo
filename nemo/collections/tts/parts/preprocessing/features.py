# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


from enum import Enum
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import torch

from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor


class TTSFeature(Enum):
    PITCH = "pitch"
    ENERGY = "energy"
    VOICED = "voiced_mask"


def compute_energy(spec: np.ndarray) -> np.ndarray:
    """
    Compute energy of the input spectrogram.

    Args:
        spec: [spec_dim, T_spec] or [B, spec_dim, T_spec] float array containing spectrogram features.

    Returns:
        [T_spec] or [B, T_spec] float array with energy value for each spectrogram frame.
    """
    energy = np.linalg.norm(spec, axis=-2)
    return energy


class PitchFeaturizer:
    def __init__(
        self,
        sample_rate: int = 22050,
        win_length: int = 1024,
        hop_length: int = 256,
        pitch_fmin: int = librosa.note_to_hz('C2'),
        pitch_fmax: int = librosa.note_to_hz('C7'),
    ) -> None:
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.pitch_fmin = pitch_fmin
        self.pitch_fmax = pitch_fmax

    def compute_pitch(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute pitch of the input audio.

        Args:
            audio: [T_audio] float array containing audio samples.

        Returns:
            pitch: [T_spec] float array containing pitch for each audio frame.
            voiced: [T_spec] bool array indicating whether each audio frame contains speech.
            voiced_prob: [T_spec] float array with [0, 1] probability that each audio frame contains speech.
        """
        pitch, voiced, voiced_prob = librosa.pyin(
            audio,
            fmin=self.pitch_fmin,
            fmax=self.pitch_fmax,
            frame_length=self.win_length,
            hop_length=self.hop_length,
            sr=self.sample_rate,
            fill_na=0.0,
        )
        return pitch, voiced, voiced_prob


class MelSpectrogramFeaturizer:
    def __init__(
        self,
        sample_rate: int = 22050,
        mel_dim: int = 80,
        win_length: int = 1024,
        hop_length: int = 256,
        lowfreq: int = 0,
        highfreq: int = 8000,
        log: bool = True,
        log_zero_guard_type: str = "add",
        log_zero_guard_value: float = 1.0,
        mel_norm: Optional[Union[str, int]] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length

        self.preprocessor = AudioToMelSpectrogramPreprocessor(
            sample_rate=sample_rate,
            features=mel_dim,
            pad_to=1,
            n_window_size=win_length,
            n_window_stride=hop_length,
            window_size=False,
            window_stride=False,
            n_fft=win_length,
            lowfreq=lowfreq,
            highfreq=highfreq,
            log=log,
            log_zero_guard_type=log_zero_guard_type,
            log_zero_guard_value=log_zero_guard_value,
            mel_norm=mel_norm,
        )

    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Computes mel spectrogram for the input audio.

        Args:
            audio: [T_audio] float array containing audio samples.

        Returns:
            [spec_dim, T_spec] float array containing spectrogram features.
        """
        # [1, T_audio]
        audio_tensor = torch.tensor(audio[np.newaxis, :], dtype=torch.float32)
        # [1]
        audio_len_tensor = torch.tensor([audio.shape[0]], dtype=torch.int32)

        # [1, spec_dim, T_spec]
        spec_tensor, _ = self.preprocessor(input_signal=audio_tensor, length=audio_len_tensor)
        # [spec_dim, T_spec]
        spec = spec_tensor.detach().numpy()[0]

        return spec
