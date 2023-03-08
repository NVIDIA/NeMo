# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List, Tuple

import torch


class TextProcessor(ABC):
    @abstractmethod
    def preprocess_text(self, text: str) -> str:
        """
        Pre-normalization text processing.

        Args:
            text: input text string.

        Returns:
            pre-processed text string.
        """
        raise NotImplementedError()

    @abstractmethod
    def normalize_text(self, text: str) -> str:
        """
        Text normalization.

        Args:
            text: input text string.

        Returns:
            normalized text string.
        """
        raise NotImplementedError()

    @abstractmethod
    def postprocess_text(self, text: str) -> str:
        """
        Post-normalization text processing.

        Args:
            text: input text string.

        Returns:
            post-processed text string.
        """
        raise NotImplementedError()


class GraphemeToPhoneme(ABC):
    @abstractmethod
    def convert_graphemes_to_phonemes(self, text: str) -> str:
        """
        Convert string containing graphemes/characters to phonemes (eg. IPA or ARPABET).
          Out of vocabulary words are left as graphemes.

        Args:
            text: post-processed text string.

        Returns:
            phoneme string
        """
        raise NotImplementedError()

    @abstractmethod
    def convert_graphemes_to_phonemes_mixed(self, text: str, phone_prob: float) -> str:
        """
        Convert each word in input string from graphemes to phonemes with probability phone_prob.
          Out of vocabulary words are left as graphemes.

        Args:
            text: post-processed text string.

        Returns:
            phoneme string
        """
        raise NotImplementedError()


class TextTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """
        Create list of integer indices for the given text string.

        Args:
            text: input text string.

        Returns:
            list of integer indices.
        """
        raise NotImplementedError()


class AudioProcessor(ABC):
    @abstractmethod
    def compute_spectrogram(self, audio: torch.tensor, audio_len: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        Computes the spectrogram of the input audio batch.

        Args:
            audio: [B, T_audio] float tensor containing [-1, 1] audio samples.
            audio_len: [B] integer tensor containing length of audio in batch.

        Returns:
            [B, spec_dim, T_spec] float tensor with spectrogram features.
            [B] integer tensor with output spectrogram lengths.
        """
        raise NotImplementedError()


class SpectrogramSynthesizer(ABC):
    @property
    @abstractmethod
    def device(self):
        pass

    @abstractmethod
    def synthesize_spectrogram(
        self, tokens: torch.tensor, speaker: torch.tensor, pitch: torch.tensor, pace: torch.tensor
    ) -> torch.tensor:
        """
        Synthesizes batch of spectrograms given an input batch of text tokens.

        Args:
            tokens: [B, T_text] integer tensor with text tokens to synthesize spectrogram for.
            speaker: [B] integer tensor with speaker ID to synthesize spectrogram for.
            pitch: [B, T_text] float tensor with adjustment for each input token.
            pace: [] scalar float tensor to control speaking rate.

        Returns:
            [B, spec_dim, T_spec] float tensor with synthesized spectrogram
        """
        raise NotImplementedError()


class Vocoder(ABC):
    @property
    @abstractmethod
    def device(self):
        pass

    @abstractmethod
    def convert_spectrogram_to_audio(self, spec: torch.tensor) -> torch.tensor:
        """
        Converts a batch of spectrograms into a batch of audio.

        Args:
            spec: [B, mel_dim, T_spec] float tensor with spectrogram data.

        Returns:
            [B, T_audio] float tensor with [1, -1] audio data.
        """
        raise NotImplementedError()
