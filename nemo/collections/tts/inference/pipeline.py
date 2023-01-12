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

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch

from nemo.utils.decorators import experimental


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
    def synthesize_spectrogram(self, tokens: torch.tensor, speaker: torch.tensor, pitch: torch.tensor) -> torch.tensor:
        """
        Synthesizes batch of spectrograms given an input batch of text tokens.

        Args:
            tokens: [B, T_text] integer tensor with text tokens to synthesize spectrogram for.
            speaker: [B] integer tensor with speaker ID to synthesize spectrogram for.
            pitch: [B, T_text] float tensor with adjustment for each input token.

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


@experimental
class TTSPipeline:
    def __init__(
        self,
        text_processor: TextProcessor = None,
        g2p: GraphemeToPhoneme = None,
        text_tokenizer: TextTokenizer = None,
        audio_processor: AudioProcessor = None,
        spectrogram_synthesizer: SpectrogramSynthesizer = None,
        vocoder: Vocoder = None,
    ):
        self.text_processor = text_processor
        self.g2p = g2p
        self.text_tokenizer = text_tokenizer
        self.audio_processor = audio_processor
        self.spectrogram_synthesizer = spectrogram_synthesizer
        self.vocoder = vocoder

    def get_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute spectrogram for input audio array.

        Args:
            audio: [T_audio]

        Returns:
            [spec_dim, T_spec] numpy array with spectrogram features
        """

        assert self.audio_preprocessor is not None
        assert len(audio.shape) == 2

        # [1, T_audio]
        audio_tensor = torch.tensor([audio], dtype=torch.float32)
        # [1]
        audio_len_tensor = torch.tensor([audio.shape[1]], dtype=torch.int32)

        # [1, spec_dim, T_spec]
        spec_tensor, _ = self.audio_processor.get_spectrogram(audio=audio_tensor, audio_len=audio_len_tensor)
        # [spec_dim, T_spec]
        spec = spec_tensor.detach().numpy()[0]

        return spec

    def process_text(self, text: str) -> str:
        """
        Run full text processing pipeline including preprocessing, normalization, and postprocessing.

        Args:
            text: input text string.

        Returns:
            post-processed text string.
        """

        assert self.text_processor is not None
        assert self.text_tokenizer is not None

        processed_text = self.text_processor.preprocess_text(text)
        processed_text = self.text_processor.normalize_text(processed_text)
        processed_text = self.text_processor.postprocess_text(processed_text)

        return processed_text

    def text_to_speech(self, text: str, speaker: int = 0, pitch: float = 0.0) -> np.ndarray:
        """
        Run full text to speech pipeline from raw inputs to final audio.

        Args:
            text: text string to synthesize.
            speaker: optional integer speaker ID of voice to synthesize, default 0.
            pitch: optional float pitch adjustment for synthesized utterance, default 0.

        Returns:
            [T_audio] numpy array containing synthesized audio samples.
        """

        assert self.spectrogram_synthesizer is not None
        assert self.vocoder is not None

        processed_text = self.process_text(text)
        if self.g2p is not None:
            processed_text = self.g2p.convert_graphemes_to_phonemes(processed_text)

        tokens = self.text_tokenizer.tokenize(processed_text)
        # [1, T_text]
        token_tensor = torch.tensor([tokens], dtype=torch.int32).to(self.spectrogram_synthesizer.device)
        # [1]
        speaker_tensor = torch.tensor([speaker], dtype=torch.int32).to(self.spectrogram_synthesizer.device)
        # [1, T_text]
        pitch_tensor = torch.tensor([len(tokens) * [pitch]], dtype=torch.float32).to(
            self.spectrogram_synthesizer.device
        )

        # [1, spec_dim, T_spec]
        spectrogram_tensor = self.spectrogram_synthesizer.synthesize_spectrogram(
            tokens=token_tensor, speaker=speaker_tensor, pitch=pitch_tensor
        )
        # [1, T_audio]
        audio_tensor = self.vocoder.convert_spectrogram_to_audio(spec=spectrogram_tensor)
        # [T_audio]
        audio = audio_tensor.detach().numpy()[0]

        return audio
