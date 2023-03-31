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


from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import librosa
import numpy as np
import torch

from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    get_audio_paths_from_manifest,
    get_feature_filename_from_manifest,
)


class Featurizer(ABC):
    @abstractmethod
    def setup(self, feature_dir: Path) -> None:
        """
        Setup machine for saving features to disk.

        Args:
            feature_dir: base directory where features will be stored.
        """

    @abstractmethod
    def compute_feature(self, manifest_entry: dict, audio_dir: Path) -> List[Union[np.ndarray, torch.tensor]]:
        """
        Compute feature value for the given manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is stored.

        Returns:
            List of feature arrays or tensors
        """

    @abstractmethod
    def save(self, manifest_entry: dict, audio_dir: Path, feature_dir: Path) -> None:
        """
        Save feature value to disk for given manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is stored.
            feature_dir: base directory where features will be stored.
        """

    @abstractmethod
    def load(self, manifest_entry: dict, audio_dir: Path, feature_dir: Path) -> List[torch.tensor]:
        """
        Read saved feature value for given manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is stored.
            feature_dir: base directory where features were stored by save().

        Returns:
            List of feature tensors
        """


class MelSpectrogramFeaturizer(Featurizer):
    def __init__(
        self,
        feature_name: str = "mel_spec",
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
        self.feature_name = feature_name
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

    def setup(self, feature_dir: Path) -> None:
        spec_dir = feature_dir / self.feature_name
        spec_dir.mkdir(exist_ok=True, parents=True)

    def compute_feature(self, manifest_entry: dict, audio_dir: Path) -> List[torch.tensor]:
        """
        Computes mel spectrogram for the input manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is store

        Returns:
            Single element list with [spec_dim, T_spec] float tensor containing spectrogram features.
        """

        audio_path, _ = get_audio_paths_from_manifest(manifest_entry=manifest_entry, audio_dir=audio_dir)
        audio, _ = librosa.load(path=audio_path, sr=self.sample_rate)

        # [1, T_audio]
        audio_tensor = torch.tensor(audio[np.newaxis, :], dtype=torch.float32)
        # [1]
        audio_len_tensor = torch.tensor([audio.shape[0]], dtype=torch.int32)

        # [1, spec_dim, T_spec]
        spec_tensor, _ = self.preprocessor(input_signal=audio_tensor, length=audio_len_tensor)
        # [spec_dim, T_spec]
        spec_tensor = spec_tensor.detach()[0]

        return [spec_tensor]

    def save(self, manifest_entry: dict, audio_dir: Path, feature_dir: Path) -> None:
        spec_tensor = self.compute_feature(manifest_entry=manifest_entry, audio_dir=audio_dir)[0]
        feature_filename = get_feature_filename_from_manifest(manifest_entry=manifest_entry, audio_dir=audio_dir)

        spec_path = feature_dir / self.feature_name / feature_filename
        torch.save(spec_tensor, spec_path)

    def load(self, manifest_entry: dict, audio_dir: Path, feature_dir: Path) -> List[torch.tensor]:
        feature_filename = get_feature_filename_from_manifest(manifest_entry=manifest_entry, audio_dir=audio_dir)
        spec_path = feature_dir / self.feature_name / feature_filename
        spec_tensor = torch.load(spec_path)

        return [spec_tensor]


class EnergyFeaturizer(Featurizer):
    def __init__(self, spec_featurizer: MelSpectrogramFeaturizer, feature_name: str = "energy",) -> None:
        self.feature_name = feature_name
        self.spec_featurizer = spec_featurizer

    def setup(self, feature_dir: Path) -> None:
        energy_dir = feature_dir / self.feature_name
        energy_dir.mkdir(exist_ok=True, parents=True)

    def compute_feature(self, manifest_entry: dict, audio_dir: Path) -> List[torch.tensor]:
        """
        Computes energy for the input manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is store

        Returns:
            Single element list with [T_spec] float tensor containing energy features.
        """
        # [1, T_audio]
        spec = self.spec_featurizer.compute_feature(manifest_entry=manifest_entry, audio_dir=audio_dir)[0]
        # [T_audio]
        energy = torch.linalg.norm(spec, axis=0)

        return [energy]

    def save(self, manifest_entry: dict, audio_dir: Path, feature_dir: Path) -> None:
        energy_tensor = self.compute_feature(manifest_entry=manifest_entry, audio_dir=audio_dir)[0]
        feature_filename = get_feature_filename_from_manifest(manifest_entry=manifest_entry, audio_dir=audio_dir)

        energy_path = feature_dir / self.feature_name / feature_filename
        torch.save(energy_tensor, energy_path)

    def load(self, manifest_entry: dict, audio_dir: Path, feature_dir: Path) -> List[torch.tensor]:
        feature_filename = get_feature_filename_from_manifest(manifest_entry=manifest_entry, audio_dir=audio_dir)
        energy_path = feature_dir / self.feature_name / feature_filename
        energy_tensor = torch.load(energy_path)

        return [energy_tensor]


class PitchFeaturizer(Featurizer):
    def __init__(
        self,
        pitch_feature_name: str = "pitch",
        voiced_feature_name: str = "voiced_mask",
        sample_rate: int = 22050,
        win_length: int = 1024,
        hop_length: int = 256,
        pitch_fmin: int = librosa.note_to_hz('C2'),
        pitch_fmax: int = librosa.note_to_hz('C7'),
    ) -> None:
        self.pitch_feature_name = pitch_feature_name
        self.voiced_feature_name = voiced_feature_name
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.pitch_fmin = pitch_fmin
        self.pitch_fmax = pitch_fmax

    def setup(self, feature_dir: Path) -> None:
        if self.pitch_feature_name:
            pitch_dir = feature_dir / self.pitch_feature_name
            pitch_dir.mkdir(exist_ok=True, parents=True)

        if self.voiced_feature_name:
            voiced_dir = feature_dir / self.voiced_feature_name
            voiced_dir.mkdir(exist_ok=True, parents=True)

    def compute_feature(self, manifest_entry: dict, audio_dir: Path) -> List[np.ndarray]:
        """
        Computes pitch and optional voiced mask for the input manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is store

        Returns:
            Three element list [[T_spec], [T_spec], [T_spec]] with pitch float array, voiced mask boolean array,
                and voiced probability float array.
        """
        audio_path, _ = get_audio_paths_from_manifest(manifest_entry=manifest_entry, audio_dir=audio_dir)
        audio, _ = librosa.load(path=audio_path, sr=self.sample_rate)

        pitch, voiced, voiced_prob = librosa.pyin(
            audio,
            fmin=self.pitch_fmin,
            fmax=self.pitch_fmax,
            frame_length=self.win_length,
            hop_length=self.hop_length,
            sr=self.sample_rate,
            fill_na=0.0,
        )
        return [pitch, voiced, voiced_prob]

    def save(self, manifest_entry: dict, audio_dir: Path, feature_dir: Path) -> None:
        pitch, voiced, _ = self.compute_feature(manifest_entry=manifest_entry, audio_dir=audio_dir)
        feature_filename = get_feature_filename_from_manifest(manifest_entry=manifest_entry, audio_dir=audio_dir)

        if self.pitch_feature_name:
            pitch_path = feature_dir / self.pitch_feature_name / feature_filename
            pitch_tensor = torch.tensor(pitch, dtype=torch.float32)
            torch.save(pitch_tensor, pitch_path)

        if self.voiced_feature_name:
            voiced_path = feature_dir / self.voiced_feature_name / feature_filename
            voiced_tensor = torch.tensor(voiced, dtype=torch.bool)
            torch.save(voiced_tensor, voiced_path)

    def load(self, manifest_entry: dict, audio_dir: Path, feature_dir: Path) -> List[torch.tensor]:
        feature_filename = get_feature_filename_from_manifest(manifest_entry=manifest_entry, audio_dir=audio_dir)

        output_features = []

        if self.pitch_feature_name:
            pitch_path = feature_dir / self.pitch_feature_name / feature_filename
            pitch = torch.load(pitch_path)
            output_features.append(pitch)

        if self.voiced_feature_name:
            voiced_path = feature_dir / self.voiced_feature_name / feature_filename
            voiced = torch.load(voiced_path)
            output_features.append(voiced)

        return output_features
