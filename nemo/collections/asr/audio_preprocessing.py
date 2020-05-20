# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
# =============================================================================
"""
This file contains neural modules responsible for preprocessing audio data.
"""
__all__ = [
    'AudioPreprocessing',
    'AudioPreprocessor',
    'AudioToMFCCPreprocessor',
    'AudioToMelSpectrogramPreprocessor',
    'AudioToSpectrogramPreprocessor',
    'CropOrPadSpectrogramAugmentation',
    'MultiplyBatch',
    'SpectrogramAugmentation',
    'TimeStretchAugmentation',
]

import math
from abc import abstractmethod

import numpy as np
import torch
from packaging import version

from .parts.features import FilterbankFeatures
from .parts.spectr_augment import SpecAugment, SpecCutout
from nemo.backends.pytorch import NonTrainableNM
from nemo.core import Optimization
from nemo.core.neural_types import *
from nemo.utils import logging
from nemo.utils.decorators import add_port_docs

try:
    import torchaudio
    import torchaudio.transforms
    import torchaudio.functional

    TORCHAUDIO_VERSION = version.parse(torchaudio.__version__)
    TORCHAUDIO_VERSION_MIN = version.parse('0.5')

    HAVE_TORCHAUDIO = True
except ModuleNotFoundError:
    HAVE_TORCHAUDIO = False
    logging.warning('Could not import torchaudio. Some features might not work.')

try:
    from apex import amp
except (AttributeError, ModuleNotFoundError) as e:
    logging.warning("Unable to import APEX. Mixed precision and distributed training will not work.")


class AudioPreprocessor(NonTrainableNM):
    """
    A base class for Neural Modules that performs audio preprocessing,
    transforming the wav files to features.
    """

    def __init__(self, win_length, hop_length):
        super().__init__()

        self.win_length = win_length
        self.hop_length = hop_length

        self.disable_casts = self._opt_level == Optimization.mxprO1

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
        if self.disable_casts:
            with amp.disable_casts():
                processed_signal = self.get_features(input_signal.to(torch.float), length)
        else:
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
        n_window_size (int): Size of window for fft in samples
            Defaults to None. Use one of window_size or n_window_size.
        n_window_stride (int): Stride of window for fft in samples
            Defaults to None. Use one of window_stride or n_window_stride.
        n_fft (int): Length of FT window. If None, it uses the smallest power
            of 2 that is larger than n_window_size.
            Defaults to None
        window (str): Windowing function for fft. can be one of ['hann',
            'hamming', 'blackman', 'bartlett', 'none', 'null']
            Defaults to "hann"
        normalized (bool): Whether to normalize by magnitude after stft
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "input_signal": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "length": NeuralType({0: AxisType(BatchTag)}),
            "input_signal": NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # "processed_signal": NeuralType(
            #    {0: AxisType(BatchTag), 1: AxisType(SpectrogramSignalTag), 2: AxisType(ProcessedTimeTag),}
            # ),
            # "processed_length": NeuralType({0: AxisType(BatchTag)}),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        sample_rate=16000,
        window_size=0.02,
        window_stride=0.01,
        n_window_size=None,
        n_window_stride=None,
        n_fft=None,
        window="hann",
        normalized=True,
    ):
        self._sample_rate = sample_rate
        if not HAVE_TORCHAUDIO:
            raise ModuleNotFoundError(
                "torchaudio is not installed but is necessary for "
                "AudioToSpectrogramPreprocessor. We recommend you try "
                "building it from source for the PyTorch version you have."
            )
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

        super().__init__(n_window_size, n_window_stride)

        self.win_length = n_window_size
        self.hop_length = n_window_stride

        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        # Set window_fn. None defaults to torch.ones.
        window_fn = self.torch_windows.get(window, None)
        if window_fn is None:
            raise ValueError(
                f"Window argument for AudioProcessor is invalid: {window}."
                f"For no window function, use 'ones' or None."
            )

        # Create featurizer.
        # Calls torch.stft under the hood, and is hard-coded to use center=True
        self.featurizer = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window_fn=window_fn,
            normalized=normalized,
        )
        self.featurizer.to(self._device)

    def get_features(self, input_signal, length):
        return self.featurizer(input_signal)

    @property
    def sample_rate(self):
        return self._sample_rate


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

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "input_signal": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "length": NeuralType({0: AxisType(BatchTag)}),
            "input_signal": NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.

        processed_signal:

            0: AxisType(BatchTag)

            1: AxisType(MelSpectrogramSignalTag)

            2: AxisType(ProcessedTimeTag)

        processed_length:

            0: AxisType(BatchTag)

        """
        return {
            # "processed_signal": NeuralType(
            #    {0: AxisType(BatchTag), 1: AxisType(MelSpectrogramSignalTag), 2: AxisType(ProcessedTimeTag),}
            # ),
            # "processed_length": NeuralType({0: AxisType(BatchTag)}),
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

        super().__init__(n_window_size, n_window_stride)

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
        self.featurizer.to(self._device)

    def get_features(self, input_signal, length):
        return self.featurizer(input_signal, length)

    def get_seq_len(self, seq_len):
        return self.featurizer.get_seq_len(seq_len)

    @property
    def filter_banks(self):
        return self.featurizer.filter_banks


class AudioToMFCCPreprocessor(AudioPreprocessor):
    """Preprocessor that converts wavs to MFCCs.
    Uses torchaudio.transforms.MFCC.

    Args:
        sample_rate: The sample rate of the audio.
            Defaults to 16000.
        window_size: Size of window for fft in seconds. Used to calculate the
            win_length arg for mel spectrogram.
            Defaults to 0.02
        window_stride: Stride of window for fft in seconds. Used to caculate
            the hop_length arg for mel spect.
            Defaults to 0.01
        n_window_size: Size of window for fft in samples
            Defaults to None. Use one of window_size or n_window_size.
        n_window_stride: Stride of window for fft in samples
            Defaults to None. Use one of window_stride or n_window_stride.
        window: Windowing function for fft. can be one of ['hann',
            'hamming', 'blackman', 'bartlett', 'none', 'null'].
            Defaults to 'hann'
        n_fft: Length of FT window. If None, it uses the smallest power of 2
            that is larger than n_window_size.
            Defaults to None
        lowfreq (int): Lower bound on mel basis in Hz.
            Defaults to 0
        highfreq  (int): Lower bound on mel basis in Hz.
            Defaults to None
        n_mels: Number of mel filterbanks.
            Defaults to 64
        n_mfcc: Number of coefficients to retain
            Defaults to 64
        dct_type: Type of discrete cosine transform to use
        norm: Type of norm to use
        log: Whether to use log-mel spectrograms instead of db-scaled.
            Defaults to True.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "input_signal": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "length": NeuralType({0: AxisType(BatchTag)}),
            "input_signal": NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # "processed_signal": NeuralType(
            #    {0: AxisType(BatchTag), 1: AxisType(MFCCSignalTag), 2: AxisType(ProcessedTimeTag),}
            # ),
            # "processed_length": NeuralType({0: AxisType(BatchTag)}),
            "processed_signal": NeuralType(('B', 'D', 'T'), MFCCSpectrogramType()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(
        self,
        sample_rate=16000,
        window_size=0.02,
        window_stride=0.01,
        n_window_size=None,
        n_window_stride=None,
        window='hann',
        n_fft=None,
        lowfreq=0.0,
        highfreq=None,
        n_mels=64,
        n_mfcc=64,
        dct_type=2,
        norm='ortho',
        log=True,
    ):
        self._sample_rate = sample_rate
        if not HAVE_TORCHAUDIO:
            raise ModuleNotFoundError(
                "torchaudio is not installed but is necessary for "
                "AudioToMFCCPreprocessor. We recommend you try "
                "building it from source for the PyTorch version you have."
            )
        if window_size and n_window_size:
            raise ValueError(f"{self} received both window_size and " f"n_window_size. Only one should be specified.")
        if window_stride and n_window_stride:
            raise ValueError(
                f"{self} received both window_stride and " f"n_window_stride. Only one should be specified."
            )
        # Get win_length (n_window_size) and hop_length (n_window_stride)
        if window_size:
            n_window_size = int(window_size * self._sample_rate)
        if window_stride:
            n_window_stride = int(window_stride * self._sample_rate)

        super().__init__(n_window_size, n_window_stride)

        mel_kwargs = {}

        mel_kwargs['f_min'] = lowfreq
        mel_kwargs['f_max'] = highfreq
        mel_kwargs['n_mels'] = n_mels

        mel_kwargs['n_fft'] = n_fft or 2 ** math.ceil(math.log2(n_window_size))

        mel_kwargs['win_length'] = n_window_size
        mel_kwargs['hop_length'] = n_window_stride

        # Set window_fn. None defaults to torch.ones.
        window_fn = self.torch_windows.get(window, None)
        if window_fn is None:
            raise ValueError(
                f"Window argument for AudioProcessor is invalid: {window}."
                f"For no window function, use 'ones' or None."
            )
        mel_kwargs['window_fn'] = window_fn

        # Use torchaudio's implementation of MFCCs as featurizer
        self.featurizer = torchaudio.transforms.MFCC(
            sample_rate=self._sample_rate,
            n_mfcc=n_mfcc,
            dct_type=dct_type,
            norm=norm,
            log_mels=log,
            melkwargs=mel_kwargs,
        )
        self.featurizer.to(self._device)

    def get_features(self, input_signal, length):
        return self.featurizer(input_signal)


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

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "input_spec": NeuralType({0: AxisType(BatchTag), 1: AxisType(SpectrogramSignalTag), 2: AxisType(
            # TimeTag),})
            "input_spec": NeuralType(('B', 'D', 'T'), SpectrogramType())
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # "augmented_spec": NeuralType(
            #    {0: AxisType(BatchTag), 1: AxisType(SpectrogramSignalTag), 2: AxisType(ProcessedTimeTag),}
            # )
            "augmented_spec": NeuralType(('B', 'D', 'T'), SpectrogramType())
        }

    def __init__(
        self,
        freq_masks=0,
        time_masks=0,
        freq_width=10,
        time_width=10,
        rect_masks=0,
        rect_time=5,
        rect_freq=20,
        rng=None,
    ):
        super().__init__()

        if rect_masks > 0:
            self.spec_cutout = SpecCutout(rect_masks=rect_masks, rect_time=rect_time, rect_freq=rect_freq, rng=rng,)
            self.spec_cutout.to(self._device)
        else:
            self.spec_cutout = lambda x: x

        if freq_masks + time_masks > 0:
            self.spec_augment = SpecAugment(
                freq_masks=freq_masks, time_masks=time_masks, freq_width=freq_width, time_width=time_width, rng=rng,
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

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "in_x": NeuralType({0: AxisType(BatchTag), 1: AxisType(SpectrogramSignalTag), 2: AxisType(TimeTag),}),
            # "in_x_len": NeuralType({0: AxisType(BatchTag)}),
            # "in_y": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "in_y_len": NeuralType({0: AxisType(BatchTag)}),
            "in_x": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "in_x_len": NeuralType(tuple('B'), LengthsType()),
            "in_y": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "in_y_len": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # "out_x": NeuralType({0: AxisType(BatchTag), 1: AxisType(SpectrogramSignalTag), 2: AxisType(TimeTag),}),
            # "out_x_len": NeuralType({0: AxisType(BatchTag)}),
            # "out_y": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "out_y_len": NeuralType({0: AxisType(BatchTag)}),
            "out_x": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "out_x_len": NeuralType(tuple('B'), LengthsType()),
            "out_y": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "out_y_len": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, mult_batch=1):
        super().__init__()
        self.mult = mult_batch

    @torch.no_grad()
    def forward(self, in_x, in_x_len, in_y, in_y_len):
        out_x = in_x.repeat(self.mult, 1, 1)
        out_y = in_y.repeat(self.mult, 1)
        out_x_len = in_x_len.repeat(self.mult)
        out_y_len = in_y_len.repeat(self.mult)

        return out_x, out_x_len, out_y, out_y_len


class CropOrPadSpectrogramAugmentation(NonTrainableNM):
    """
    Pad or Crop the incoming Spectrogram to a certain shape.

    Args:
        audio_length (int): the final number of timesteps that is required.
            The signal will be either padded or cropped temporally to this
            size.
    """

    def __init__(self, audio_length, **kwargs):
        super(CropOrPadSpectrogramAugmentation, self).__init__()
        self.audio_length = audio_length

    @torch.no_grad()
    def forward(self, input_signal, length):
        image = input_signal
        num_images = image.shape[0]

        audio_length = self.audio_length
        image_len = image.shape[-1]

        # Crop long signal
        if image_len > audio_length:  # randomly slice
            cutout_images = []
            offset = torch.randint(low=0, high=image_len - audio_length + 1, size=[num_images])

            # TODO: Look into advanced broadcasting to speed up section
            for idx, offset in enumerate(offset):
                cutout_images.append(image[idx : idx + 1, :, offset : offset + audio_length])

            image = torch.cat(cutout_images, dim=0)
            del cutout_images

        else:  # symmetrically pad short signal with zeros
            pad_left = (audio_length - image_len) // 2
            pad_right = (audio_length - image_len) // 2

            if (audio_length - image_len) % 2 == 1:
                pad_right += 1

            image = torch.nn.functional.pad(image, [pad_left, pad_right], mode="constant", value=0)

        # Replace dynamic length sequences with static number of timesteps
        length = (length * 0) + audio_length

        return image, length

    @property
    def input_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # "input_signal": NeuralType(
            #     {0: AxisType(BatchTag), 1: AxisType(SpectrogramSignalTag), 2: AxisType(ProcessedTimeTag), }
            # ),
            # "length": NeuralType({0: AxisType(BatchTag)}),
            "input_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            # "processed_signal": NeuralType(
            #     {0: AxisType(BatchTag), 1: AxisType(SpectrogramSignalTag), 2: AxisType(ProcessedTimeTag), }
            # ),
            # "processed_length": NeuralType({0: AxisType(BatchTag)}),
            "processed_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }


class TimeStretchAugmentation(NonTrainableNM):
    def __init__(
        self,
        sample_rate: int,
        probability: float,
        min_speed_rate: float = 0.9,
        max_speed_rate: float = 1.1,
        num_rates: int = 5,
        n_fft: int = 512,
    ):
        """
        Time-stretch a batch of audio series by a fixed rate while preserving pitch.

        Note that while the speed rate is sampled independently for every batch,
        all samples of that batch will be augmented by the same speed rate.

        Note:
        This is a simplified implementation, intended primarily for reference and pedagogical purposes.
        It makes no attempt to handle transients, and is likely to produce audible artifacts.

        Args:
            sample_rate: Sampling rate.
            probability: Float value declaring chance of the input being augmented.
                Must be a float value in the range [0, 1].
            min_speed_rate: Minimum sampling rate modifier.
            max_speed_rate: Maximum sampling rate modifier.
            num_rates: Number of discrete rates to allow. Can be a positive or negative
                integer.
                If a positive integer greater than 0 is provided, the range of
                speed rates will be discretized into `num_rates` values.
                If a negative integer or 0 is provided, the full range of speed rates
                will be sampled uniformly.
                Note: If a positive integer is provided and the resultant discretized
                range of rates contains the value '1.0', then those samples with rate=1.0,
                will not be augmented at all and simply skipped. This is to avoid unnecessary
                augmentation and increase computation time. Effective augmentation chance
                in such a case is = `prob * (num_rates - 1 / num_rates) * 100`% chance
                where `prob` is the global probability of a sample being augmented.
            n_fft: Number of fft filters to be computed.
        """
        super(TimeStretchAugmentation, self).__init__()

        if probability > 1.0 or probability < 0.0:
            raise ValueError("`probability` must be between 0 and 1")

        if not HAVE_TORCHAUDIO:
            raise ModuleNotFoundError(
                "torchaudio is not installed but is necessary for "
                "TimeStretchAugmentation. We recommend you try "
                "installing it from conda for the PyTorch version you have."
            )

        # Check torchaudio version; inform user of potential issue
        if TORCHAUDIO_VERSION < TORCHAUDIO_VERSION_MIN:
            logging.error(
                "Current installed version of `torchaudio` %s is less than the recommended minimum "
                "version of %s. Please note that this may cause deadlocks when using distributed "
                "data parallel training. Please follow the instructions at https://github.com/pytorch/audio "
                "to update torchaudio.",
                str(TORCHAUDIO_VERSION),
                str(TORCHAUDIO_VERSION_MIN),
            )

        min_rate = min(min_speed_rate, max_speed_rate)
        if min_rate < 0.0:
            raise ValueError("Minimum sampling rate modifier must be > 0.")

        self._sample_rate = sample_rate
        self.probability = float(probability)
        self.min_rate = float(min_speed_rate)
        self.max_rate = float(max_speed_rate)
        self.num_rates = num_rates
        if num_rates > 0:
            self._rates = np.linspace(min_speed_rate, max_speed_rate, num_rates)
        self._rng = np.random.RandomState()

        self._n_fft = n_fft
        self._hop_length = n_fft // 2
        self._stft_window = torch.hann_window(self._n_fft, periodic=True, device=self._device)
        self._phi_advance = torch.linspace(0, np.pi * self._hop_length, self._hop_length + 1, device=self._device)
        self._phi_advance = self._phi_advance.view(-1, 1)

    @torch.no_grad()
    def forward(self, input_signal, length):
        proba = self._rng.uniform(0.0, 1.0)

        if proba > self.probability:
            return input_signal, length

        # Select speed rate either from choice or random sample
        if self.num_rates < 0:
            speed_rate = self._rng.uniform(self.min_rate, self.max_rate)
        else:
            speed_rate = np.random.choice(self._rates)

        # Skip perturbation in case of identity speed rate
        if speed_rate == 1.0:
            return input_signal, length

        features = self._stft(input_signal, self._n_fft, self._hop_length)
        features = self._phase_vocoder(features, speed_rate)

        # Predict the length of y_stretch
        len_stretch = int(round(input_signal.shape[1] / speed_rate))

        audio = self._istft(features, len_stretch)

        length = (length * speed_rate).type(torch.long)

        return audio, length

    def _stft(self, data: torch.Tensor, n_fft: int, hop_length: int):
        win_length = n_fft
        window = self._stft_window

        stft = torch.stft(
            data,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode='reflect',
            normalized=False,
        )
        return stft

    def _phase_vocoder(self, data: torch.Tensor, rate: float):
        data_stretch = torchaudio.functional.phase_vocoder(data, rate, self._phi_advance)
        return data_stretch

    def _istft(self, data: torch.Tensor, len_stretch: int):
        n_fft = 2 * (data.shape[1] - 1)
        hop_length = self._hop_length
        win_length = n_fft
        window = self._stft_window

        audio = torchaudio.functional.istft(
            data,
            n_fft,
            hop_length,
            win_length,
            window=window,
            center=True,
            pad_mode='reflect',
            normalized=False,
            length=len_stretch,
        )

        return audio

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "input_signal": NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        return {
            "processed_signal": NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            "processed_length": NeuralType(tuple('B'), LengthsType()),
        }


def AudioPreprocessing(*args, **kwargs):
    raise NotImplementedError(
        "AudioPreprocessing has been deprecated and replaced by: "
        "AudioToMFCCPreprocessor, AudioToMelSpectrogramPreprocessor, and "
        "AudioToSpectrogramPreprocessor. For most ASR purposes "
        "AudioToMelSpectrogramPreprocessor does the same as the old "
        "AudioPreprocessing."
    )
