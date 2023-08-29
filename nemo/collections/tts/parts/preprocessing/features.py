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
import io
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
from einops import rearrange
from torch import Tensor

from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.collections.tts.parts.utils.tarred_dataset_utils import VALID_AUDIO_FORMATS
from nemo.collections.tts.parts.utils.tts_dataset_utils import (
    get_audio_filepaths,
    load_audio,
    normalize_volume,
    stack_tensors
)
from nemo.utils.decorators import experimental


def save_numpy_feature(
    feature_name: Optional[str],
    features: np.ndarray,
    manifest_entry: Dict[str, Any],
    audio_dir: Path,
    feature_dir: Path,
) -> None:
    """
    If feature_name is provided, save feature as .npy file.
    """
    if feature_name is None:
        return

    feature_filepath = _get_numpy_feature_filepath(
        manifest_entry=manifest_entry, audio_dir=audio_dir, feature_dir=feature_dir, feature_name=feature_name
    )
    feature_filepath.parent.mkdir(exist_ok=True, parents=True)
    np.save(file=str(feature_filepath), arr=features)


def _get_numpy_feature_filepath(
    manifest_entry: Dict[str, Any], audio_dir: Path, feature_dir: Path, feature_name: str
) -> Path:
    """
    Get the absolute path for the feature file corresponding to the input manifest entry

    Example: audio_filepath "<audio_dir>/speaker1/audio1.wav" becomes
        feature_filepath "<feature_dir>/<feature_name>/speaker1/audio1.npy"
    """
    _, audio_filepath_rel = get_audio_filepaths(manifest_entry=manifest_entry, audio_dir=audio_dir)
    feature_filepath = feature_dir / feature_name / audio_filepath_rel.with_suffix(".npy")
    return feature_filepath


def _features_exists(
    feature_names: List[Optional[str]], manifest_entry: Dict[str, Any], audio_dir: Path, feature_dir: Path,
) -> bool:
    for feature_name in feature_names:
        if feature_name is None:
            continue
        feature_filepath = _get_numpy_feature_filepath(
            manifest_entry=manifest_entry, audio_dir=audio_dir, feature_dir=feature_dir, feature_name=feature_name
        )
        if not feature_filepath.exists():
            return False
    return True


def _get_frame_indices(manifest_entry: Dict[str, Any], sample_rate: int, hop_length: int) -> Optional[Tuple[int, int]]:
    if "offset" not in manifest_entry:
        return None

    offset = manifest_entry["offset"]
    duration = manifest_entry["duration"]
    start_i = librosa.core.time_to_frames(offset, sr=sample_rate, hop_length=hop_length)
    end_i = 1 + start_i + librosa.core.time_to_frames(duration, sr=sample_rate, hop_length=hop_length)
    return start_i, end_i


def _feature_collate_fn(feature_name: str, train_batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
    """
    Combine list/batch of features into a feature dictionary.
    """
    feature_dict = {}
    feature_tensors = []
    for example in train_batch:
        feature_array = example[feature_name]
        feature_tensor = torch.from_numpy(feature_array)
        feature_tensors.append(feature_tensor)

    max_len = max([f.shape[0] for f in feature_tensors])
    stacked_features = stack_tensors(feature_tensors, max_lens=[max_len])
    feature_dict[feature_name] = stacked_features

    return feature_dict


@experimental
class Featurizer(ABC):
    @abstractmethod
    def save(self, manifest_entry: Dict[str, Any], audio_dir: Path, feature_dir: Path, overwrite: bool = True) -> None:
        """
        Save feature value to disk for given manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is stored.
            feature_dir: base directory where features will be stored.
            overwrite: whether to overwrite features if they already exist.
        """
        pass


@experimental
class FeatureReader(ABC):

    def __init__(self, feature_name: str):
        self.feature_name = feature_name

    @abstractmethod
    def get_feature_filepath(
        self, manifest_entry: Dict[str, Any], audio_dir: Path, feature_dir: Path
    ) -> Path:
        pass

    @abstractmethod
    def get_tarred_suffix(self, feature_filepath: Path) -> str:
        pass

    @abstractmethod
    def get_tarred_suffixes(self) -> str:
        pass

    @abstractmethod
    def serialize(self, manifest_entry: Dict[str, Any], audio_dir: Path, feature_dir: Path) -> io.BytesIO:
        """
        Serialize feature file into bytes.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is stored.
            feature_dir: base directory where features are stored.

        Returns:
            Bytes io with feature file contents.
        """
        pass

    @abstractmethod
    def deserialize(self, bytes: io.BytesIO) -> Dict[str, np.ndarray]:
        """
        Deserialize feature bytes into a dictionary.

        Args:
            bytes: bytes IO

        Returns:
            Bytes io containing feature file contents.
        """
        pass

    @abstractmethod
    def load(self, manifest_entry: Dict[str, Any], audio_dir: Path, feature_dir: Path) -> Dict[str, np.ndarray]:
        """
        Read feature file into a dictionary.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is stored.
            feature_dir: base directory where features are stored.

        Returns:
            Feature dictionary.
        """
        pass

    @abstractmethod
    def collate_fn(self, train_batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        """
        Combine batch of features into a feature dictionary.
        """
        pass


class NumpyFeatureReader(FeatureReader):

    def __init__(
        self, feature_name: str, sample_rate: int = 22050, hop_length: int = 256, feature_filename: Optional[str] = None
    ):
        super().__init__(feature_name=feature_name)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        if feature_filename:
            self.feature_filename = feature_filename
        else:
            self.feature_filename = feature_name

    def get_feature_filepath(
        self, manifest_entry: Dict[str, Any], audio_dir: Path, feature_dir: Path
    ) -> Path:
        return _get_numpy_feature_filepath(
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
            feature_name=self.feature_filename
        )

    def get_tarred_suffix(self, feature_filepath: Optional[Path]) -> str:
        suffix = f"{self.feature_name}.npy"
        return suffix

    def get_tarred_suffixes(self) -> str:
        return self.get_tarred_suffix(None)

    def serialize(self, manifest_entry: Dict[str, Any], audio_dir: Path, feature_dir: Path) -> io.BytesIO:
        feature_dict = self.load(manifest_entry=manifest_entry, audio_dir=audio_dir, feature_dir=feature_dir)
        feature_array = feature_dict[self.feature_name]
        bytes_io = io.BytesIO()
        np.save(file=bytes_io, arr=feature_array)
        return bytes_io

    def deserialize(self, bytes_io: io.BytesIO) -> Dict[str, np.ndarray]:
        feature_array = np.load(file=bytes_io, allow_pickle=True)
        feature_dict = {self.feature_name: feature_array}
        return feature_dict

    def load(self, manifest_entry: Dict[str, Any], audio_dir: Path, feature_dir: Path) \
        -> Dict[str, np.ndarray]:
        """
        Read saved feature value for given manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is stored.
            feature_dir: base directory where features are stored.

        Returns:
            Feature array
        """
        indices = _get_frame_indices(
            manifest_entry=manifest_entry, sample_rate=self.sample_rate, hop_length=self.hop_length
        )

        feature_filepath = self.get_feature_filepath(
            manifest_entry=manifest_entry, audio_dir=audio_dir, feature_dir=feature_dir
        )
        feature_filepath = str(feature_filepath)

        if indices:
            feature_mmap = np.load(feature_filepath, mmap_mode='r')
            feature_array = feature_mmap[indices[0]: indices[1]]
            feature_array = np.copy(feature_array)
        else:
            feature_array = np.load(feature_filepath)

        feature_dict = {self.feature_name: feature_array}

        return feature_dict

    def collate_fn(self, train_batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        return _feature_collate_fn(feature_name=self.feature_name, train_batch=train_batch)


class AudioFeatureReader(FeatureReader):

    def __init__(
        self,
        feature_name="audio",
        feature_len_name: str = "audio_lens",
        sample_rate: int = 22050,
        volume_norm: bool = False
    ):
        super().__init__(feature_name=feature_name)
        self.feature_len_name = feature_len_name
        self.sample_rate = sample_rate
        self.volume_norm = volume_norm
        self.valid_formats = VALID_AUDIO_FORMATS

    def get_feature_filepath(
        self, manifest_entry: Dict[str, Any], audio_dir: Path, feature_dir: Path
    ) -> Path:
        audio_filepath, _ = get_audio_filepaths(manifest_entry=manifest_entry, audio_dir=audio_dir)
        return audio_filepath

    def get_tarred_suffix(self, feature_filepath: Path) -> str:
        # Use extension of input file without preceding "."
        return feature_filepath.suffix[1:]

    def get_tarred_suffixes(self) -> str:
        return self.valid_formats

    def serialize(self, manifest_entry: Dict[str, Any], audio_dir: Path, feature_dir: Path) -> io.BytesIO:
        audio, _, _ = load_audio(
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            sample_rate=self.sample_rate,
            volume_norm=False,
        )
        bytes_io = io.BytesIO()
        # Fill in file name for bytes IO so that soundfile knows the file format.
        bytes_io.name = manifest_entry["audio_filepath"]
        sf.write(file=bytes_io, data=audio, samplerate=self.sample_rate)
        return bytes_io

    def deserialize(self, bytes_io: io.BytesIO) -> Dict[str, np.ndarray]:
        audio, sample_rate = sf.read(file=bytes_io, dtype='float32')
        audio_len = audio.shape[0]

        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Serialized audio has wrong sample rate. Expected {self.sample_rate}, found {sample_rate}"
            )

        if self.volume_norm:
            audio = normalize_volume(audio)

        feature_dict = {
            self.feature_name: audio,
            self.feature_len_name: audio_len
        }

        return feature_dict

    def load(self, manifest_entry: Dict[str, Any], audio_dir: Path, feature_dir: Path) \
        -> Dict[str, np.ndarray]:
        """
        Read saved feature value for given manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is stored.
            feature_dir: base directory where features are stored.

        Returns:
            Feature array
        """
        audio, _, audio_filepath_rel = load_audio(
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            sample_rate=self.sample_rate,
            volume_norm=self.volume_norm,
        )
        audio_len = audio.shape[0]
        feature_dict = {
            self.feature_name: audio,
            self.feature_len_name: audio_len
        }

        return feature_dict

    def collate_fn(self, train_batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        audio_list = []
        audio_len_list = []
        for example in train_batch:
            audio_array = example[self.feature_name]
            audio_len = example[self.feature_len_name]
            audio_tensor = torch.from_numpy(audio_array)
            audio_list.append(audio_tensor)
            audio_len_list.append(audio_len)

        batch_audio_len = torch.IntTensor(audio_len_list)
        audio_max_len = int(batch_audio_len.max().item())
        batch_audio = stack_tensors(audio_list, max_lens=[audio_max_len])

        batch_dict = {
            self.feature_name: batch_audio,
            self.feature_len_name: batch_audio_len,
        }

        return batch_dict


class AudioCodecFeatureReader(NumpyFeatureReader):

    def __init__(
        self,
        feature_name: str = "audio_tokens",
        feature_filename: str = "audio_tokens",
        feature_len_field: str = "audio_token_lens",
        sample_rate: int = 22050,
        hop_length: int = 256
    ):
        super().__init__(
            feature_name=feature_name, feature_filename=feature_filename, sample_rate=sample_rate, hop_length=hop_length
        )
        self.feature_len_field = feature_len_field

    def deserialize(self, bytes_io: io.BytesIO) -> Dict[str, np.ndarray]:
        feature_dict = super().deserialize(bytes_io)
        feature_dict[self.feature_len_field] = feature_dict[self.feature_name].shape[0]
        return feature_dict

    def load(self, manifest_entry: Dict[str, Any], audio_dir: Path, feature_dir: Path) \
        -> Dict[str, np.ndarray]:
        """
        Read saved feature value for given manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is stored.
            feature_dir: base directory where features are stored.

        Returns:
            Feature array
        """
        feature_dict = super().load(manifest_entry=manifest_entry, audio_dir=audio_dir, feature_dir=feature_dir)
        feature_dict[self.feature_len_field] = feature_dict[self.feature_name].shape[0]
        return feature_dict

    def collate_fn(self, train_batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        token_list = []
        token_len_list = []
        for example in train_batch:
            token_array = example[self.feature_name]
            token_len = example[self.feature_len_field]
            token_tensor = torch.from_numpy(token_array)
            token_tensor = rearrange(token_tensor, 'T C -> C T')
            token_list.append(token_tensor)
            token_len_list.append(token_len)

        batch_token_len = torch.IntTensor(token_len_list)
        token_max_len = int(batch_token_len.max().item())
        batch_tokens = stack_tensors(token_list, max_lens=[token_max_len])

        batch_dict = {
            self.feature_name: batch_tokens,
            self.feature_len_field: batch_token_len,
        }

        return batch_dict


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
        volume_norm: bool = True,
    ) -> None:
        self.feature_name = feature_name
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.volume_norm = volume_norm

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
            mag_power=1.0,
            log=log,
            log_zero_guard_type=log_zero_guard_type,
            log_zero_guard_value=log_zero_guard_value,
            mel_norm=mel_norm,
            normalize=None,
            preemph=None,
            dither=0.0,
        )

    def compute_mel_spec(self, manifest_entry: Dict[str, Any], audio_dir: Path) -> np.ndarray:
        """
        Computes mel spectrogram for the input manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is store

        Returns:
            [spec_dim, T_spec] float tensor containing spectrogram features.
        """
        audio_filepath_abs, _ = get_audio_filepaths(manifest_entry=manifest_entry, audio_dir=audio_dir)
        audio, _ = librosa.load(audio_filepath_abs, sr=self.sample_rate)

        if self.volume_norm:
            audio = normalize_volume(audio)

        # [1, T_audio]
        audio_tensor = torch.tensor(audio[np.newaxis, :], dtype=torch.float32)
        # [1]
        audio_len_tensor = torch.tensor([audio.shape[0]], dtype=torch.int32)

        # [1, spec_dim, T_spec]
        spec_tensor, _ = self.preprocessor(input_signal=audio_tensor, length=audio_len_tensor)
        # [spec_dim, T_spec]
        spec_tensor = spec_tensor.detach()[0]
        spec_array = spec_tensor.numpy()

        return spec_array

    def save(self, manifest_entry: Dict[str, Any], audio_dir: Path, feature_dir: Path, overwrite: bool = True) -> None:
        if not overwrite and _features_exists(
            feature_names=[self.feature_name],
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
        ):
            return

        spec = self.compute_mel_spec(manifest_entry=manifest_entry, audio_dir=audio_dir)
        save_numpy_feature(
            feature_name=self.feature_name,
            features=spec,
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
        )


class EnergyFeaturizer(Featurizer):
    def __init__(self, spec_featurizer: MelSpectrogramFeaturizer, feature_name: str = "energy") -> None:
        self.feature_name = feature_name
        self.spec_featurizer = spec_featurizer

    def compute_energy(self, manifest_entry: Dict[str, Any], audio_dir: Path) -> np.ndarray:
        """
        Computes energy for the input manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is store

        Returns:
            [T_spec] float tensor containing energy features.
        """
        # [spec_dim, T_spec]
        spec = self.spec_featurizer.compute_mel_spec(manifest_entry=manifest_entry, audio_dir=audio_dir)
        # [T_spec]
        energy = np.linalg.norm(spec, axis=0)
        return energy

    def save(self, manifest_entry: Dict[str, Any], audio_dir: Path, feature_dir: Path, overwrite: bool = True) -> None:
        if not overwrite and _features_exists(
            feature_names=[self.feature_name],
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
        ):
            return

        energy = self.compute_energy(manifest_entry=manifest_entry, audio_dir=audio_dir)
        save_numpy_feature(
            feature_name=self.feature_name,
            features=energy,
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
        )


class PitchFeaturizer(Featurizer):
    """
    Class for computing pitch features.

    Args:
        pitch_name: Optional directory name to save pitch features under.
            If None, then pitch will not be saved.
        voiced_mask_name: Optional directory name to save voiced mask under.
            If None, then voiced mask will not be saved.
        voiced_prob_name: Optional directory name to save voiced probabilities under.
            If None, then voiced probabilities will not be saved.
        sample_rate: Sample rate to use when loading audio.
        win_length: Audio frame length to use for pitch computation.
        hop_length: Audio hop length to use for pitch computation.
        pitch_fmin: Minimum pitch value to compute. Defaults to librosa.note_to_hz('C2') = 65.41 Hz.
        pitch_fmax: Maximum pitch value to compute. Defaults to librosa.note_to_hz('C7') = 2093.00 Hz.
            Setting this to a lower value will speed up computation, but may lose some pitch information.
        volume_norm: Whether to apply volume normalization to the audio.
        batch_seconds: Optional float, if provided then long audio files will have their pitch computed after
            splitting them into segments batch_seconds seconds long, to avoid running out of memory.
        batch_padding: If batch_seconds is provided, then this determines how many audio frames will be padded on
            both sides of each segment to ensure that the pitch values at the boundary are correct.
            If batch_seconds is not provided then this parameter is ignored.
    """

    def __init__(
        self,
        pitch_name: Optional[str] = "pitch",
        voiced_mask_name: Optional[str] = "voiced_mask",
        voiced_prob_name: Optional[str] = None,
        sample_rate: int = 22050,
        win_length: int = 1024,
        hop_length: int = 256,
        pitch_fmin: int = librosa.note_to_hz('C2'),
        pitch_fmax: int = librosa.note_to_hz('C7'),
        volume_norm: bool = True,
        batch_seconds: Optional[float] = 30.0,
        batch_padding: int = 10,
    ) -> None:
        self.pitch_name = pitch_name
        self.voiced_mask_name = voiced_mask_name
        self.voiced_prob_name = voiced_prob_name
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.volume_norm = volume_norm
        self.pitch_fmin = pitch_fmin
        self.pitch_fmax = pitch_fmax
        if batch_seconds:
            assert batch_padding is not None
            # Round sample size up to a multiple of hop_length
            batch_samples = int(batch_seconds * sample_rate)
            self.batch_frames = int(math.ceil(batch_samples / self.hop_length))
            self.batch_samples = self.hop_length * self.batch_frames
            self.batch_padding_frames = batch_padding
            self.batch_padding_samples = self.hop_length * self.batch_padding_frames
        else:
            self.batch_samples = None
            self.batch_padding = None

    def compute_pitch(
        self, manifest_entry: Dict[str, Any], audio_dir: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes pitch and optional voiced mask for the input manifest entry.

        Args:
            manifest_entry: Manifest entry dictionary.
            audio_dir: base directory where audio is store

        Returns:
            pitch: [T_spec] float tensor containing pitch for each audio frame.
            voiced_mask: [T_spec] bool tensor indicating whether each audio frame is voiced.
            voiced_prob: [T_spec] float array with [0, 1] probability that each audio frame is voiced.
        """
        audio_filepath_abs, _ = get_audio_filepaths(manifest_entry=manifest_entry, audio_dir=audio_dir)
        audio, _ = librosa.load(audio_filepath_abs, sr=self.sample_rate)

        if self.volume_norm:
            audio = normalize_volume(audio)

        if not self.batch_samples or audio.shape[0] < self.batch_samples:
            pitch, voiced_mask, voiced_prob = librosa.pyin(
                audio,
                fmin=self.pitch_fmin,
                fmax=self.pitch_fmax,
                sr=self.sample_rate,
                frame_length=self.win_length,
                hop_length=self.hop_length,
                fill_na=0.0,
            )
        else:
            num_chunks = int(np.ceil(audio.shape[0] / self.batch_samples))
            pitch_list = []
            voiced_mask_list = []
            voiced_prob_list = []

            for i in range(num_chunks):
                start_i = i * self.batch_samples
                end_i = (i + 1) * self.batch_samples

                if i != 0:
                    # Pad beginning with additional frames
                    start_i -= self.batch_padding_samples
                if i != (num_chunks - 1):
                    # Pad end with additional frames
                    end_i += self.batch_padding_samples

                audio_chunk = audio[start_i:end_i]
                pitch_i, voiced_mask_i, voiced_prob_i = librosa.pyin(
                    audio_chunk,
                    fmin=self.pitch_fmin,
                    fmax=self.pitch_fmax,
                    sr=self.sample_rate,
                    frame_length=self.win_length,
                    hop_length=self.hop_length,
                    fill_na=0.0,
                )
                # Remove padded frames
                if i != 0:
                    pitch_i = pitch_i[self.batch_padding_frames :]
                    voiced_mask_i = voiced_mask_i[self.batch_padding_frames :]
                    voiced_prob_i = voiced_prob_i[self.batch_padding_frames :]
                if i != (num_chunks - 1):
                    pitch_i = pitch_i[: self.batch_frames]
                    voiced_mask_i = voiced_mask_i[: self.batch_frames]
                    voiced_prob_i = voiced_prob_i[: self.batch_frames]

                pitch_list.append(pitch_i)
                voiced_mask_list.append(voiced_mask_i)
                voiced_prob_list.append(voiced_prob_i)

            pitch = np.concatenate(pitch_list, axis=0)
            voiced_mask = np.concatenate(voiced_mask_list, axis=0)
            voiced_prob = np.concatenate(voiced_prob_list, axis=0)

        pitch = pitch.astype(np.float32)
        voiced_prob = voiced_prob.astype(np.float32)

        return pitch, voiced_mask, voiced_prob

    def save(self, manifest_entry: Dict[str, Any], audio_dir: Path, feature_dir: Path, overwrite: bool = True) -> None:
        if not overwrite and _features_exists(
            feature_names=[self.pitch_name, self.voiced_mask_name, self.voiced_prob_name],
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
        ):
            return

        pitch, voiced_mask, voiced_prob = self.compute_pitch(manifest_entry=manifest_entry, audio_dir=audio_dir)
        save_numpy_feature(
            feature_name=self.pitch_name,
            features=pitch,
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
        )
        save_numpy_feature(
            feature_name=self.voiced_mask_name,
            features=voiced_mask,
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
        )
        save_numpy_feature(
            feature_name=self.voiced_prob_name,
            features=voiced_prob,
            manifest_entry=manifest_entry,
            audio_dir=audio_dir,
            feature_dir=feature_dir,
        )
