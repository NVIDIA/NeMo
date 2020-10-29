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

from nemo.collections.tts.models import *  # Avoid circular imports
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo


class SpectrogramGenerator(ModelPT, ABC):
    """ Base class for all TTS models that turn text into a spectrogram """

    @abstractmethod
    def parse(self, str_input: str, **kwargs) -> 'torch.tensor':
        """
        A helper function that accepts raw python strings and turns it into a tensor. The tensor should have 2
        dimensions. The first is the batch, which should be of size 1. The second should represent time. The tensor
        should represent either tokenized or embedded text, depending on the model.
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


class TextToWaveform(ModelPT, ABC):
    """ Base class for all end-to-end TTS models that generate a waveform from text """

    @abstractmethod
    def parse(self, str_input: str, **kwargs) -> 'torch.tensor':
        """
        A helper function that accepts raw python strings and turns it into a tensor. The tensor should have 2
        dimensions. The first is the batch, which should be of size 1. The second should represent time. The tensor
        should represent either tokenized or embedded text, depending on the model.
        """

    @abstractmethod
    def convert_text_to_waveform(self, str_input: str, **kwargs) -> 'torch.tensor':
        """
        Accepts a batch of text and returns a batch of audio

        Args:
            tokens: A torch tensor representing the text to be converted to speech

        Returns:
            audio: A torch tensor representing the waveform output.
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
