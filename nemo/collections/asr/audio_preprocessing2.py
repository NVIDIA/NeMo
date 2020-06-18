# -*- coding: utf-8 -*-

# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,


# limitations under the License.
# =============================================================================
from abc import abstractmethod

# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
from typing import Any, Dict

import torch

from nemo.collections.asr.parts.features import FilterbankFeatures
from nemo.core import AudioSignal, LengthsType, MelSpectrogramType, NeuralType
from nemo.core.apis import NeuralModulePT

__all__ = ['AudioToMelSpectrogramPreprocessor2']


class AudioPreprocessor2(NeuralModulePT):
    """
    A base class for Neural Modules that performs audio preprocessing,
    transforming the wav files to features.
    """

    def __init__(self, win_length, hop_length):
        super().__init__()

        self.win_length = win_length
        self.hop_length = hop_length

        self.torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'ones': torch.ones,
            None: torch.ones,
        }

    @torch.no_grad()
    def forward(self, input_signal, length):
        processed_signal = self.get_features(input_signal, length)
        processed_length = self.get_seq_len(length.float())

        return processed_signal, processed_length

    @abstractmethod
    def get_features(self, input_signal, length):
        # Called by forward(). Subclasses should implement this.
        pass

    def get_seq_len(self, length):
        # Called by forward()
        return torch.ceil(length / self.hop_length).to(dtype=torch.long)


class AudioToMelSpectrogramPreprocessor2(AudioPreprocessor2):
    """Featurizer that converts wavs to mel spectrograms.
    We don't use torchaudio's implementation here because the original
    implementation is not the same, so for the sake of backwards-compatibility
    this will use the old FilterbankFeatures for now.

    Args:
        sample_rate (int): Sample rate of the input audio data.
            Defaults to 16000
        window_size (float): Size of window for fft in seconds
            Defaults to 0.02
        window_stride (float): Stride of window for fft in seconds
            Defaults to 0.01
        n_window_size (int): Size of window for fft in samples
            Defaults to None. Use one of window_size or n_window_size.
        n_window_stride (int): Stride of window for fft in samples
            Defaults to None. Use one of window_stride or n_window_stride.
        window (str): Windowing function for fft. can be one of ['hann',
            'hamming', 'blackman', 'bartlett']
            Defaults to "hann"
        normalize (str): Can be one of ['per_feature', 'all_features']; all
            other options disable feature normalization. 'all_features'
            normalizes the entire spectrogram to be mean 0 with std 1.
            'pre_features' normalizes per channel / freq instead.
            Defaults to "per_feature"
        n_fft (int): Length of FT window. If None, it uses the smallest power
            of 2 that is larger than n_window_size.
            Defaults to None
        preemph (float): Amount of pre emphasis to add to audio. Can be
            disabled by passing None.
            Defaults to 0.97
        features (int): Number of mel spectrogram freq bins to output.
            Defaults to 64
        lowfreq (int): Lower bound on mel basis in Hz.
            Defaults to 0
        highfreq  (int): Lower bound on mel basis in Hz.
            Defaults to None
        log (bool): Log features.
            Defaults to True
        log_zero_guard_type(str): Need to avoid taking the log of zero. There
            are two options: "add" or "clamp".
            Defaults to "add".
        log_zero_guard_value(float, or str): Add or clamp requires the number
            to add with or clamp to. log_zero_guard_value can either be a float
            or "tiny" or "eps". torch.finfo is used if "tiny" or "eps" is
            passed.
            Defaults to 2**-24.
        dither (float): Amount of white-noise dithering.
            Defaults to 1e-5
        pad_to (int): Ensures that the output size of the time dimension is
            a multiple of pad_to.
            Defaults to 16
        frame_splicing (int): Defaults to 1
        stft_conv (bool): If True, uses pytorch_stft and convolutions. If
            False, uses torch.stft.
            Defaults to False
        pad_value (float): The value that shorter mels are padded with.
            Defaults to 0
        mag_power (float): The power that the linear spectrogram is raised to
            prior to multiplication with mel basis.
            Defaults to 2 for a power spec
    """

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    @classmethod
    def from_config_dict(cls, configuration: Dict[str, Any]):
        raise NotImplementedError()

    def to_config_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "input_signal": NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.

        processed_signal:

            0: AxisType(BatchTag)

            1: AxisType(MelSpectrogramSignalTag)

            2: AxisType(ProcessedTimeTag)

        processed_length:

            0: AxisType(BatchTag)

        """
        return {
            "processed_signal": NeuralType(('B', 'D', 'T'), MelSpectrogramType()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        sample_rate=16000,
        window_size=0.02,
        window_stride=0.01,
        n_window_size=None,
        n_window_stride=None,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        features=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2 ** -24,
        dither=1e-5,
        pad_to=16,
        frame_splicing=1,
        stft_conv=False,
        pad_value=0,
        mag_power=2.0,
    ):
        super(AudioToMelSpectrogramPreprocessor2, self).__init__(n_window_size, n_window_stride)

        self._sample_rate = sample_rate
        if window_size and n_window_size:
            raise ValueError(f"{self} received both window_size and " f"n_window_size. Only one should be specified.")
        if window_stride and n_window_stride:
            raise ValueError(
                f"{self} received both window_stride and " f"n_window_stride. Only one should be specified."
            )
        if window_size:
            n_window_size = int(window_size * self._sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * self._sample_rate)

        self.featurizer = FilterbankFeatures(
            sample_rate=self._sample_rate,
            n_window_size=n_window_size,
            n_window_stride=n_window_stride,
            window=window,
            normalize=normalize,
            n_fft=n_fft,
            preemph=preemph,
            nfilt=features,
            lowfreq=lowfreq,
            highfreq=highfreq,
            log=log,
            log_zero_guard_type=log_zero_guard_type,
            log_zero_guard_value=log_zero_guard_value,
            dither=dither,
            pad_to=pad_to,
            frame_splicing=frame_splicing,
            stft_conv=stft_conv,
            pad_value=pad_value,
            mag_power=mag_power,
        )

    def get_features(self, input_signal, length):
        return self.featurizer(input_signal, length)

    def get_seq_len(self, seq_len):
        return self.featurizer.get_seq_len(seq_len)

    @property
    def filter_banks(self):
        return self.featurizer.filter_banks
