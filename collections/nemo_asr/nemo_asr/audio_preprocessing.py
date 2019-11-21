# Copyright (C) NVIDIA CORPORATION. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.****
"""
This file contains neural modules responsible for preprocessing audio data.
"""
__all__ = ['AudioPreprocessor',
           'AudioToMFCCPreprocessor',
           'AudioToMelSpectrogramPreprocessor',
           'AudioToSpectrogramPreprocessor',
           'MultiplyBatch',
           'SpectrogramAugmentation']

import math
import torch
import torchaudio
try:
    from apex import amp
except AttributeError:
    print("Unable to import APEX. Mixed precision and distributed training "
          "will not work.")

from nemo.backends.pytorch import NonTrainableNM
from nemo.core import Optimization
from nemo.core.neural_types import *
from .parts.features import FilterbankFeatures
from .parts.spectr_augment import SpecAugment, SpecCutout


class AudioPreprocessor(NonTrainableNM):
    """
    A base class for Neural Modules that performs audio preprocessing,
    transforming the wav files to features.
    """
    @staticmethod
    def create_ports():
        input_ports = {
            "input_signal": NeuralType({0: AxisType(BatchTag),
                                        1: AxisType(TimeTag)}),

            "length": NeuralType({0: AxisType(BatchTag)}),
        }

        output_ports = {
            "processed_signal": NeuralType({0: AxisType(BatchTag),
                                            1: AxisType(SpectrogramSignalTag),
                                            2: AxisType(ProcessedTimeTag)}),

            "processed_length": NeuralType({0: AxisType(BatchTag)})
        }
        return input_ports, output_ports

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.disable_casts = (self._opt_level == Optimization.mxprO1)

        self.torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }

    @torch.no_grad()
    def forward(self, input_signal, length):
        if self.disable_casts:
            with amp.disable_casts():
                processed_signal = self.get_features(
                        input_signal.to(torch.float), length)
        else:
            processed_signal = self.get_features(input_signal, length)

        processed_length = self.get_seq_len(length)
        return processed_signal, processed_length

    def get_features(self, input_signal, length):
        raise NotImplementedError

    def get_seq_len(self, length):
        raise NotImplementedError


class AudioToSpectrogramPreprocessor(AudioPreprocessor):
    """Preprocessor that converts wavs to spectrograms.
    Uses torchaudio's Spectrogram class as a featurizer.

    Args:
        sample_rate (int): Sample rate of the input audio data.
            Defaults to 16000
        window_size (float): Size of window for fft in seconds
            Defaults to 0.02
        window_stride (float): Stride of window for fft in seconds
            Defaults to 0.01
        n_fft (int): Length of FT window. If None, it uses the smallest power
            of 2 that is larger than n_window_size.
            Defaults to None
        window (str): Windowing function for fft. can be one of ['hann',
            'hamming', 'blackman', 'bartlett']
            Defaults to "hamming"
        normalized (bool): Whether to normalize by magnitude after stft
        pad (int): Two sided padding
    """
    def __init__(
            self, *,
            sample_rate=16000,
            window_size=0.02,
            window_stride=0.01,
            n_fft=None,
            window="hamming",
            normalized=True,
            pad=8
    ):
        super().__init__(**kwargs)

        self.win_length = int(sample_rate * window_size)
        self.hop_length = int(sample_rate * window_stride)
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        window_fn = self.torch_windows.get(window, None)

        # Create featurizer
        self.featurizer = torchaudio.transform.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=pad,
            window_fn=window_fn,
            normalized=normalized
        )
        self.featurizer.to(self._device)

    def get_features(self, input_signal, length):
        return self.featurizer(input_signal)

    def get_seq_len(self, seq_len):
        return torch.ceil(
                seq_len.to(dtype=torch.float) / self.hop_length).to(
                        dtype=torch.long)


class AudioToMelSpectrogramPreprocessor(AudioPreprocessor):
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
        dither (float): Amount of white-noise dithering.
            Defaults to 1e-5
        pad_to (int): Ensures that the output size of the time dimension is
            a multiple of pad_to.
            Defaults to 16
        frame_splicing (int): Defaults to 1
        stft_conv (bool): If True, uses pytorch_stft and convolutions. If
            False, uses torch.stft.
            Defaults to False
    """
    def __init__(
            self, *,
            sample_rate=16000,
            window_size=0.02,
            window_stride=0.01,
            window="hann",
            normalize="per_feature",
            n_fft=None,
            preemph=0.97,
            features=64,
            lowfreq=0,
            highfreq=None,
            log=True,
            dither=1e-5,
            pad_to=16,
            frame_splicing=1,
            stft_conv=False,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.featurizer = FilterbankFeatures(
            sample_rate=sample_rate,
            window_size=window_size,
            window_stride=window_stride,
            window=window,
            normalize=normalize,
            n_fft=n_fft,
            preemph=preemph,
            nfilt=features,
            lowfreq=lowfreq,
            highfreq=highfreq,
            log=log,
            dither=dither,
            pad_to=pad_to,
            frame_splicing=frame_splicing,
            stft_conv=stft_conv,
            logger=self._logger
        )
        self.featurizer.to(self._device)

    def get_features(self, input_signal, length):
        return self.featurizer(input_signal, length)

    def get_seq_len(self, seq_len):
        return self.featurizer.get_seq_len(seq_len)


class AudioToMFCCPreprocessor(AudioPreprocessor):
    """Preprocessor that converts wavs to MFCCs.
    Uses torchaudio.transforms.MFCC, and several arguments are the same.

    Args:
        sample_rate: The sample rate of the audio.
            Defaults to 16000.
        window_size: Size of window for fft in seconds. Used to calculate the
            win_length arg for mel spectrogram, overrides mel_kwargs if set.
            Defaults to None
        window_stride: Stride of window for fft in seconds. Used to caculate
            the hop_length arg for mel spect, overrides mel_kwargs if set.
            Defaults to None
        window: Windowing function for fft. can be one of ['hann',
            'hamming', 'blackman', 'bartlett', None]. Overrides mel_kwargs
            if set; otherwise, torchaudio defaults to hann.
            Defaults to None
        n_mfcc: Number of coefficients to retain
            Defaults to 64
        dct_type: Type of discrete cosine transform to use
        norm: Type of norm to use
        log: Whether to use log-mel spectrograms instead of db-scaled.
            Defaults to True.
        mel_kwargs: Dict of arguments for torchaudio.transforms.MelSpectrogram
    """
    def __init__(
            self, *,
            sample_rate=16000,
            window_size=None,
            window_stride=None,
            window=None,
            n_mfcc=64,
            dct_type=2,
            norm='ortho',
            log=True,
            mel_kwargs=None,
            **kwargs):
        super().__init__(**kwargs)

        if mel_kwargs is None:
            mel_kwargs = {}

        # Use the sample rate given instead of in mel_kwargs
        if 'sample_rate' in mel_kwargs:
            del mel_kwargs['sample_rate']

        # Override mel_kwargs if window_size or window_stride are set
        if window_size is not None:
            mel_kwargs['win_length'] = int(sample_rate * window_size)
        if window_stride is not None:
            mel_kwargs['hop_length'] = int(sample_rate * window_stride)

        # Set window_fn if window arg is given
        if window:
            mel_kwargs['window_fn'] = self.torch_windows.get(window, None)

        # Use torchaudio's implementation of MFCCs as featurizer
        self.featurizer = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            dct_type=dct_type,
            norm=norm,
            log_mels=log,
            melkwargs=mel_kwargs
        )
        self.featurizer.to(self._device)

    def get_features(self, input_signal, length):
        return self.featurizer(input_signal)

    def get_seq_len(self, seq_len):
        return torch.ceil(
                seq_len.to(dtype=torch.float) /
                self.featurizer.MelSpectrogram.hop_length).to(dtype=torch.long)


class SpectrogramAugmentation(NonTrainableNM):
    """
    Performs time and freq cuts in one of two ways.

    SpecAugment zeroes out vertical and horizontal sections as described in
    SpecAugment (https://arxiv.org/abs/1904.08779). Arguments for use with
    SpecAugment are `freq_masks`, `time_masks`, `freq_width`, and `time_width`.

    SpecCutout zeroes out rectangulars as described in Cutout
    (https://arxiv.org/abs/1708.04552). Arguments for use with Cutout are
    `rect_masks`, `rect_freq`, and `rect_time`.

    Args:
        freq_masks (int): how many frequency segments should be cut.
            Defaults to 0.
        time_masks (int): how many time segments should be cut
            Defaults to 0.
        freq_width (int): maximum number of frequencies to be cut in one
            segment.
            Defaults to 10.
        time_width (int): maximum number of time steps to be cut in one
            segment
            Defaults to 10.
        rect_masks (int): how many rectangular masks should be cut
            Defaults to 0.
        rect_freq (int): maximum size of cut rectangles along the frequency
            dimension
            Defaults to 5.
        rect_time (int): maximum size of cut rectangles along the time
            dimension
            Defaults to 25.
    """
    @staticmethod
    def create_ports():
        input_ports = {
            "input_spec": NeuralType({0: AxisType(BatchTag),
                                      1: AxisType(SpectrogramSignalTag),
                                      2: AxisType(TimeTag)})
        }

        output_ports = {
            "augmented_spec": NeuralType({0: AxisType(BatchTag),
                                          1: AxisType(SpectrogramSignalTag),
                                          2: AxisType(ProcessedTimeTag)})
        }
        return input_ports, output_ports

    def __init__(
            self, *,
            freq_masks=0,
            time_masks=0,
            freq_width=10,
            time_width=10,
            rect_masks=0,
            rect_time=5,
            rect_freq=20,
            rng=None,
            **kwargs
    ):
        NonTrainableNM.__init__(self, **kwargs)

        if rect_masks > 0:
            self.spec_cutout = SpecCutout(
                rect_masks=rect_masks,
                rect_time=rect_time,
                rect_freq=rect_freq,
                rng=rng
            )
            self.spec_cutout.to(self._device)
        else:
            self.spec_cutout = lambda x: x

        if freq_masks + time_masks > 0:
            self.spec_augment = SpecAugment(
                freq_masks=freq_masks,
                time_masks=time_masks,
                freq_width=freq_width,
                time_width=time_width,
                rng=rng
            )
            self.spec_augment.to(self._device)
        else:
            self.spec_augment = lambda x: x

    def forward(self, input_spec):
        augmented_spec = self.spec_cutout(input_spec)
        augmented_spec = self.spec_augment(augmented_spec)
        return augmented_spec


class MultiplyBatch(NonTrainableNM):
    """
    Augmentation that repeats each element in a batch.
    Other augmentations can be applied afterwards.

    Args:
        mult_batch (int): number of repeats
    """
    @staticmethod
    def create_ports():
        input_ports = {
            "in_x": NeuralType({0: AxisType(BatchTag),
                                1: AxisType(SpectrogramSignalTag),
                                2: AxisType(TimeTag)}),

            "in_x_len": NeuralType({0: AxisType(BatchTag)}),

            "in_y": NeuralType({0: AxisType(BatchTag),
                                1: AxisType(TimeTag)}),

            "in_y_len": NeuralType({0: AxisType(BatchTag)})
        }

        output_ports = {
            "out_x": NeuralType({0: AxisType(BatchTag),
                                 1: AxisType(SpectrogramSignalTag),
                                 2: AxisType(TimeTag)}),

            "out_x_len": NeuralType({0: AxisType(BatchTag)}),

            "out_y": NeuralType({0: AxisType(BatchTag),
                                 1: AxisType(TimeTag)}),

            "out_y_len": NeuralType({0: AxisType(BatchTag)})
        }
        return input_ports, output_ports

    def __init__(self, *, mult_batch=1, **kwargs):
        NonTrainableNM.__init__(self, **kwargs)
        self.mult = mult_batch

    @torch.no_grad()
    def forward(self, in_x, in_x_len, in_y, in_y_len):
        out_x = in_x.repeat(self.mult, 1, 1)
        out_y = in_y.repeat(self.mult, 1)
        out_x_len = in_x_len.repeat(self.mult)
        out_y_len = in_y_len.repeat(self.mult)

        return out_x, out_x_len, out_y, out_y_len
