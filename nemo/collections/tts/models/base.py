# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from contextlib import ExitStack, contextmanager

import torch
from torch_stft import STFT

from nemo.collections.common.parts.patch_utils import istft_patch, stft_patch
from nemo.collections.tts.helpers.helpers import OperationMode
from nemo.collections.tts.models import *  # Avoid circular imports
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import AudioSignal
from nemo.core.neural_types.neural_type import NeuralType


class SpectrogramGenerator(ModelPT, ABC):
    """ Base class for all TTS models that turn text into a spectrogram """

    @abstractmethod
    def parse(self, str_input: str, **kwargs) -> 'torch.tensor':
        """
        A helper function that accepts raw pythong strings and turns it into a tensor. The tensor should have 2
        dimensions. The first is the batch, which should be of size 1. The second should represent time. The tensor
        should represented either tokenized or embedded text, depending on the model.
        """

    @abstractmethod
    def generate_spectrogram(self, tokens: 'torch.tensor', **kwargs) -> 'torch.tensor':
        """
        Accepts a batch of text or text_tokens and returns a batch of spectrograms

        Args:
            tokens: A torch tensor representing the text to be generated

        Returns:
            sepctrograms
        """

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        for subclass in cls.__subclasses__():
            subclass_models = subclass.list_available_models()
            if subclass_models is not None and len(subclass_models) > 0:
                list_of_models.extend(subclass_models)
        return list_of_models


class Vocoder(ModelPT, ABC):
    """ Base class for all TTS models that generate audio conditioned a on spectrogram """

    @abstractmethod
    def convert_spectrogram_to_audio(self, spec: 'torch.tensor', **kwargs) -> 'torch.tensor':
        """
        Accepts a batch of spectrograms and returns a batch of audio

        Args:
            spec: A torch tensor representing the spectrograms to be vocoded

        Returns:
            audio
        """

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        for subclass in cls.__subclasses__():
            subclass_models = subclass.list_available_models()
            if subclass_models is not None and len(subclass_models) > 0:
                list_of_models.extend(subclass_models)
        return list_of_models


class GlowVocoder(Vocoder):
    """ Base class for all Vocoders that use a Glow or reversible Flow-based setup. All child class are expected
    to have a parameter called audio_to_melspec_precessor that is an instance of
    nemo.collections.asr.parts.FilterbankFeatures"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = OperationMode.infer
        self.bias_spect = None
        self.n_fft = None
        self.hop_length = None
        self.win_length = None
        self.window = None
        self.n_mel = None

    @property
    def mode(self):
        return self._mode

    @contextmanager
    def temp_mode(self, mode):
        old_mode = self.mode
        self.mode = mode
        try:
            yield
        finally:
            self.mode = old_mode

    @contextmanager
    def nemo_infer(self):  # Prepend with nemo to avoid any .infer() clashes with lightning or pytorch
        with ExitStack() as stack:
            stack.enter_context(self.temp_mode(OperationMode.infer))
            stack.enter_context(torch.no_grad())
            yield

    def check_children_attributes(self):
        if self.n_fft is None:
            try:
                self.n_fft = self.audio_to_melspec_precessor.n_fft
                self.hop_length = self.audio_to_melspec_precessor.hop_length
                self.win_length = self.audio_to_melspec_precessor.win_length
                self.window = self.audio_to_melspec_precessor.window
                self.n_mel = self.audio_to_melspec_precessor.nfilt

            except AttributeError as e:
                # raise AttributeError(
                #     f"{self} could not find an audio_to_melspec_precessor. GlowVocoder requires child class to have"
                #     "audio_to_melspec_precessor defined to obtain stft parameters. audio_to_melspec_precessor "
                #     "requires n_fft, hop_length, win_length, window, and nfilt to be defined."
                # ) from e
                self.n_fft = 1024
                self.hop_length = 256
                self.win_length = 1024
                self.window = torch.hann_window(1024)
                self.n_mel = 80

    def update_bias_spect(self):
        self.check_children_attributes()  # Ensure stft parameters are defined

        with self.nemo_infer():
            spect = torch.zeros((1, self.n_mel, 88)).to(self.device)
            bias_audio = self.convert_spectrogram_to_audio(spec=spect, sigma=0.0, denoise=False)
            spect = stft_patch(
                bias_audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window.to(bias_audio.device),
            )
            if spect.dtype in [torch.cfloat, torch.cdouble]:
                spect = torch.view_as_real(spect)
            bias_spect = torch.sqrt(spect.pow(2).sum(-1))
            self.bias_spect = bias_spect[:, :, 0][:, :, None]

    @typecheck(
        input_types={"audio": NeuralType(('B', 'T'), AudioSignal()), "strength": NeuralType(optional=True)},
        output_types={"audio": NeuralType(('B', 'T'), AudioSignal())},
    )
    def denoise(self, audio: 'torch.tensor', strength: float = 0.01):
        self.check_children_attributes()  # Ensure self.n_mel and self.stft are defined

        if self.bias_spect is None:
            self.update_bias_spect()
        spect = stft_patch(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(audio.device),
        )
        if spect.dtype in [torch.cfloat, torch.cdouble]:
            spect = torch.view_as_real(spect)
        audio_spect = torch.sqrt(spect.pow(2).sum(-1))
        audio_angles = torch.atan2(spect[..., -1], spect[..., 0])
        audio_spect_denoised = audio_spect - self.bias_spect.to(audio.device) * strength
        audio_spect_denoised = torch.clamp(audio_spect_denoised, 0.0)
        audio_denoised = istft_patch(
            torch.complex(
                audio_spect_denoised * torch.cos(audio_angles), audio_spect_denoised * torch.sin(audio_angles)
            ),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.ones(self.win_length).to(audio_spect_denoised.device),
        )
        return audio_denoised


class LinVocoder(ModelPT, ABC):
    """ Base class for all TTS models that generate audio conditioned a on spectrogram """

    @abstractmethod
    def convert_linear_spectrogram_to_audio(self, spec: 'torch.tensor', **kwargs) -> 'torch.tensor':
        """
        Accepts a batch of linear spectrograms and returns a batch of audio

        Args:
            spec: A torch tensor representing the spectrograms to be vocoded ['B', 'n_freqs', 'T']

        Returns:
            audio
        """

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        for subclass in cls.__subclasses__():
            subclass_models = subclass.list_available_models()
            if subclass_models is not None and len(subclass_models) > 0:
                list_of_models.extend(subclass_models)
        return list_of_models


class MelToSpec(ModelPT, ABC):
    """ Base class for all TTS models that generate audio conditioned a on spectrogram """

    @abstractmethod
    def convert_mel_spectrogram_to_linear(self, mel: 'torch.tensor', **kwargs) -> 'torch.tensor':
        """
        Accepts a batch of spectrograms and returns a batch of audio

        Args:
            mel: A torch tensor representing the mel encoded spectrograms ['B', 'mel_freqs', 'T']

        Returns:
            spec: A torch tensor representing the linears spectrograms ['B', 'n_freqs', 'T']
        """

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        for subclass in cls.__subclasses__():
            subclass_models = subclass.list_available_models()
            if subclass_models is not None and len(subclass_models) > 0:
                list_of_models.extend(subclass_models)
        return list_of_models
