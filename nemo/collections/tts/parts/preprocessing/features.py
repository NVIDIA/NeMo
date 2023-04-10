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


class TorchFileFeaturizer(Featurizer):
    def __init__(self, feature_names: List[str]) -> None:
        self.feature_names = feature_names

    @abstractmethod
    def compute_features(self, manifest_entry: dict, audio_dir: Path) -> List[torch.tensor]:
        """
        Compute feature value for the given manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is stored.

        Returns:
            List of feature tensors with same length as self.feature_names.
        """

    def setup(self, feature_dir: Path) -> None:
        for feature_name in self.feature_names:
            feature_path = feature_dir / feature_name
            feature_path.mkdir(exist_ok=True, parents=True)

    def save(self, manifest_entry: dict, audio_dir: Path, feature_dir: Path) -> None:
        feature_filename = get_feature_filename_from_manifest(manifest_entry=manifest_entry, audio_dir=audio_dir)
        feature_tensors = self.compute_features(manifest_entry=manifest_entry, audio_dir=audio_dir)

        for feature_name, feature_tensor in zip(self.feature_names, feature_tensors):
            feature = feature_dir / feature_name / feature_filename
            torch.save(feature_tensor, feature)

    def load(self, manifest_entry: dict, audio_dir: Path, feature_dir: Path) -> List[torch.tensor]:
        feature_filename = get_feature_filename_from_manifest(manifest_entry=manifest_entry, audio_dir=audio_dir)
        output_tensors = []
        for feature_name in self.feature_names:
            feature_path = feature_dir / feature_name / feature_filename
            feature_tensor = torch.load(feature_path)
            output_tensors.append(feature_tensor)

        return output_tensors


class MelSpectrogramFeaturizer(TorchFileFeaturizer):
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
        super().__init__(feature_names=[feature_name])
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

    def compute_features(self, manifest_entry: dict, audio_dir: Path) -> List[torch.tensor]:
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


class EnergyFeaturizer(TorchFileFeaturizer):
    def __init__(self, spec_featurizer: MelSpectrogramFeaturizer, feature_name: str = "energy") -> None:
        super().__init__(feature_names=[feature_name])
        self.spec_featurizer = spec_featurizer

    def compute_features(self, manifest_entry: dict, audio_dir: Path) -> List[torch.tensor]:
        """
        Computes energy for the input manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is store

        Returns:
            Single element list with [T_spec] float tensor containing energy features.
        """
        # [1, T_audio]
        spec = self.spec_featurizer.compute_features(manifest_entry=manifest_entry, audio_dir=audio_dir)[0]
        # [T_audio]
        energy = torch.linalg.norm(spec, axis=0)

        return [energy]


class PitchFeaturizer(TorchFileFeaturizer):
    def __init__(
        self,
        pitch_feature_name: Optional[str] = "pitch",
        voiced_mask_feature_name: Optional[str] = "voiced_mask",
        voiced_prob_feature_name: Optional[str] = None,
        sample_rate: int = 22050,
        win_length: int = 1024,
        hop_length: int = 256,
        pitch_fmin: int = librosa.note_to_hz('C2'),
        pitch_fmax: int = librosa.note_to_hz('C7'),
    ) -> None:
        feature_names = []
        if pitch_feature_name:
            self.include_pitch = True
            feature_names.append(pitch_feature_name)
        else:
            self.include_pitch = False

        if voiced_mask_feature_name:
            self.include_voiced_mask = True
            feature_names.append(voiced_mask_feature_name)
        else:
            self.include_voiced_mask = False

        if voiced_prob_feature_name:
            self.include_voiced_prob = True
            feature_names.append(voiced_prob_feature_name)
        else:
            self.include_voiced_prob = False

        super().__init__(feature_names=feature_names)
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.pitch_fmin = pitch_fmin
        self.pitch_fmax = pitch_fmax

    def compute_features(self, manifest_entry: dict, audio_dir: Path) -> List[torch.Tensor]:
        """
        Computes pitch and optional voiced mask for the input manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is store

        Returns:
            List [[T_spec], [T_spec], [T_spec]] with optional pitch float tensor, voiced mask boolean tensor,
                and voiced probability float tensor.
        """
        audio_path, _ = get_audio_paths_from_manifest(manifest_entry=manifest_entry, audio_dir=audio_dir)
        audio, _ = librosa.load(path=audio_path, sr=self.sample_rate)

        pitch, voiced_mask, voiced_prob = librosa.pyin(
            audio,
            fmin=self.pitch_fmin,
            fmax=self.pitch_fmax,
            frame_length=self.win_length,
            hop_length=self.hop_length,
            sr=self.sample_rate,
            fill_na=0.0,
        )
        output_tensors = []
        if self.include_pitch:
            pitch_tensor = torch.tensor(pitch, dtype=torch.float32)
            output_tensors.append(pitch_tensor)

        if self.include_voiced_mask:
            voiced_mask_tensor = torch.tensor(voiced_mask, dtype=torch.bool)
            output_tensors.append(voiced_mask_tensor)

        if self.include_voiced_prob:
            voiced_prob_tensor = torch.tensor(voiced_prob, dtype=torch.float32)
            output_tensors.append(voiced_prob_tensor)

        return output_tensors
