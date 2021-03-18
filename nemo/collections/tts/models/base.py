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
    """ Base class for all Vocoders that use a Glow or reversible Flow-based setup. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = OperationMode.infer
        self.bias_spect = None
        self.stft = None  # Required to be defined in children classes
        self.n_mel = None  # Required to be defined in children classes

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
        if self.stft is None or self.n_mel is None:
            try:
                self.stft = self.audio_to_melspec_precessor.stft
                self.n_mel = self.audio_to_melspec_precessor.nfilt
            except AttributeError:
                raise AttributeError(
                    f"{self} did not have stft and n_mel defined. These two parameters are required for GlowVocoder's "
                    "methods to work"
                )

    def update_bias_spect(self):
        self.check_children_attributes()  # Ensure self.n_mel and self.stft are defined

        with self.nemo_infer():
            spect = torch.zeros((1, self.n_mel, 88)).to(self.device)
            bias_audio = self.convert_spectrogram_to_audio(spec=spect, sigma=0.0, denoise=False)
            bias_spect, _ = self.stft.transform(bias_audio)
            self.bias_spect = bias_spect[:, :, 0][:, :, None]

    @typecheck(
        input_types={"audio": NeuralType(('B', 'T'), AudioSignal()), "strength": NeuralType(optional=True)},
        output_types={"audio": NeuralType(('B', 'T'), AudioSignal())},
    )
    def denoise(self, audio: 'torch.tensor', strength: float = 0.01):
        self.check_children_attributes()  # Ensure self.n_mel and self.stft are defined

        if self.bias_spect is None:
            self.update_bias_spect()
        audio_spect, audio_angles = self.stft.transform(audio)
        audio_spect_denoised = audio_spect - self.bias_spect.to(audio.device) * strength
        audio_spect_denoised = torch.clamp(audio_spect_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spect_denoised, audio_angles)
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
