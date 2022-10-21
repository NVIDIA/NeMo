# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from typing import Tuple

import librosa
import numpy as np
import torch

from nemo.collections.asr.models import EncDecClassificationModel
from nemo.collections.tts.data.data_utils import normalize_volume
from nemo.utils import logging


class AudioTrimmer(ABC):
    """Interface for silence trimming implementations
    """

    @abstractmethod
    def trim_audio(self, audio: np.array, sample_rate: int, audio_id: str) -> Tuple[np.array, int, int]:
        """Trim starting and trailing silence from the input audio.
           Args:
               audio: Numpy array containing audio samples. Float [-1.0, 1.0] format.
               sample_rate: Sample rate of input audio.
               audio_id: String identifier (eg. file name) used for logging.

           Returns numpy array with trimmed audio, and integer sample indices representing the start and end
           of speech within the original audio array.
        """
        raise NotImplementedError


class EnergyAudioTrimmer(AudioTrimmer):
    def __init__(
        self,
        db_threshold: int = 50,
        ref_amplitude: float = 1.0,
        frame_threshold: int = 1,
        frame_length: int = 2048,
        frame_step: int = 512,
        pad_seconds: float = 0.1,
        volume_norm: bool = True,
    ):
        """Energy/power based silence trimming using Librosa backend.
           Args:
               db_threshold: Audio frames at least db_threshold decibels below ref_amplitude will be
                 considered silence.
               ref_amplitude: Amplitude threshold for classifying speech versus silence.
               frame_threshold: Start and end of speech will be detected where there are at least frame_threshold
                 consecutive audio frames classified as speech. Setting this value higher is more robust to
                 false-positives (silence detected as speech), but setting it too high may result in very short
                 speech segments being cut out from the audio.
               frame_length: Length of audio frames to use when doing speech detection. This does not need to match
                 the frame_length used any other part of the code or model.
               frame_step: Stride of audio frames to use when doing speech detection. This does not need to match
                 the frame_step used any other part of the code or model.
               pad_seconds: Amount of audio to keep before the detected start of speech and after the end of
                 speech. Set this to at least 0.1 to avoid cutting off any speech audio, with larger values
                 being safer but increasing the amount of silence left afterwards.
               volume_norm: Whether to normalize the volume of audio before doing speech detection.
        """
        self.db_threshold = db_threshold
        self.ref_amplitude = ref_amplitude
        self.frame_threshold = frame_threshold
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.pad_seconds = pad_seconds
        self.volume_norm = volume_norm

    def trim_audio(self, audio: np.array, sample_rate: int, audio_id: str = "") -> Tuple[np.array, int, int]:
        if self.volume_norm:
            # Normalize volume so we have a fixed scale relative to the reference amplitude
            audio = normalize_volume(audio=audio, volume_level=1.0)

        speech_frames = librosa.effects._signal_to_frame_nonsilent(
            audio,
            ref=self.ref_amplitude,
            frame_length=self.frame_length,
            hop_length=self.frame_step,
            top_db=self.db_threshold,
        )

        start_i, end_i = get_start_and_end_of_speech(
            is_speech=speech_frames,
            frame_threshold=self.frame_threshold,
            frame_step=self.frame_step,
            audio_id=audio_id,
        )

        start_i, end_i = pad_sample_indices(
            start_sample_i=start_i,
            end_sample_i=end_i,
            max_sample=audio.shape[0],
            sample_rate=sample_rate,
            pad_seconds=self.pad_seconds,
        )

        trimmed_audio = audio[start_i:end_i]

        return trimmed_audio, start_i, end_i


class VadAudioTrimmer(AudioTrimmer):
    def __init__(
        self,
        model_name: str = "vad_multilingual_marblenet",
        vad_sample_rate: int = 16000,
        vad_threshold: float = 0.4,
        device: str = "cpu",
        frame_threshold: int = 1,
        frame_length: int = 2048,
        frame_step: int = 512,
        pad_seconds: float = 0.1,
        volume_norm: bool = True,
    ):
        """Voice activity detection (VAD) based silence trimming.

           Args:
               model_name: NeMo VAD model to load. Valid configurations can be found with
                 EncDecClassificationModel.list_available_models()
               vad_sample_rate: Sample rate used for pretrained VAD model.
               vad_threshold: Softmax probability [0, 1] of VAD output, above which audio frames will be classified
                 as speech.
               device: Device "cpu" or "cuda" to use for running the VAD model.
               frame_length: Length of audio frames to use when doing speech detection. This does not need to match
                 the frame_length used any other part of the code or model.
               frame_step: Stride of audio frames to use when doing speech detection. This does not need to match
                 the frame_step used any other part of the code or model.
               pad_seconds: Amount of audio to keep before the detected start of speech and after the end of
                 speech. Set this to at least 0.1 to avoid cutting off any speech audio, with larger values
                 being safer but increasing the amount of silence left afterwards.
               volume_norm: Whether to normalize the volume of audio before doing speech detection.
        """
        self.device = device
        self.vad_model = EncDecClassificationModel.from_pretrained(model_name=model_name).eval().to(self.device)
        self.vad_sample_rate = vad_sample_rate
        self.vad_threshold = vad_threshold

        self.frame_threshold = frame_threshold
        self.frame_length = frame_length
        self.frame_step = frame_step

        self.pad_seconds = pad_seconds
        self.volume_norm = volume_norm

    def _detect_speech(self, audio: np.array) -> np.array:
        # Center-pad the audio
        audio = np.pad(audio, [self.frame_length // 2, self.frame_length // 2])

        # [num_frames, frame_length]
        audio_frames = librosa.util.frame(
            audio, frame_length=self.frame_length, hop_length=self.frame_step
        ).transpose()

        num_frames = audio_frames.shape[0]
        # [num_frames, frame_length]
        audio_signal = torch.tensor(audio_frames, dtype=torch.float32, device=self.device)
        # [1]
        audio_signal_len = torch.tensor(num_frames * [self.frame_length], dtype=torch.int32, device=self.device)

        # VAD outputs 2 values for each audio frame with logits indicating the likelihood that
        # each frame is non-speech or speech, respectively.
        # [num_frames, 2]
        log_probs = self.vad_model(input_signal=audio_signal, input_signal_length=audio_signal_len)
        probs = torch.softmax(log_probs, dim=-1)
        probs = probs.cpu().detach().numpy()
        # [num_frames]
        speech_probs = probs[:, 1]
        speech_frames = speech_probs >= self.vad_threshold

        return speech_frames

    def _scale_sample_indices(self, start_sample_i: int, end_sample_i: int, sample_rate: int) -> Tuple[int, int]:
        sample_rate_ratio = sample_rate / self.vad_sample_rate
        start_sample_i = sample_rate_ratio * start_sample_i
        end_sample_i = sample_rate_ratio * end_sample_i
        return start_sample_i, end_sample_i

    def trim_audio(self, audio: np.array, sample_rate: int, audio_id: str = "") -> Tuple[np.array, int, int]:
        if sample_rate == self.vad_sample_rate:
            vad_audio = audio
        else:
            # Downsample audio to match sample rate of VAD model
            vad_audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.vad_sample_rate)

        if self.volume_norm:
            # Normalize volume so we have a fixed scale relative to the reference amplitude
            vad_audio = normalize_volume(audio=vad_audio, volume_level=1.0)

        speech_frames = self._detect_speech(audio=vad_audio)

        start_i, end_i = get_start_and_end_of_speech(
            is_speech=speech_frames,
            frame_threshold=self.frame_threshold,
            frame_step=self.frame_step,
            audio_id=audio_id,
        )

        if sample_rate != self.vad_sample_rate:
            # Convert sample indices back to input sample rate
            start_i, end_i = self._scale_sample_indices(start_i, end_i, sample_rate)

        start_i, end_i = pad_sample_indices(
            start_sample_i=start_i,
            end_sample_i=end_i,
            max_sample=audio.shape[0],
            sample_rate=sample_rate,
            pad_seconds=self.pad_seconds,
        )

        trimmed_audio = audio[start_i:end_i]

        return trimmed_audio, start_i, end_i


def get_start_and_end_of_speech(
    is_speech: np.array, frame_threshold: int, frame_step: int, audio_id: str = ""
) -> Tuple[int, int]:
    """Finds the start and end of speech for an utterance.
       Args:
           is_speech: [num_frames] boolean array with true entries labeling speech frames.
           frame_threshold: The number of consecutive speech frames required to classify the speech boundaries.
           frame_step: Audio frame stride used to covert frame boundaries to audio samples.
           audio_id: String identifier (eg. file name) used for logging.

       Returns integers representing the sample indicies of the start and of speech.
    """
    num_frames = is_speech.shape[0]

    # Iterate forwards over the utterance until we find the first frame_threshold consecutive speech frames.
    start_i = None
    for i in range(0, num_frames):
        high_i = min(num_frames, i + frame_threshold)
        if all(is_speech[i:high_i]):
            start_i = i
            break

    # Iterate backwards over the utterance until we find the last frame_threshold consecutive speech frames.
    end_i = None
    for i in range(num_frames, 0, -1):
        low_i = max(0, i - frame_threshold)
        if all(is_speech[low_i:i]):
            end_i = i
            break

    if start_i is None:
        logging.warning(f"Could not find start of speech for '{audio_id}'")
        start_i = 0

    if end_i is None:
        logging.warning(f"Could not find end of speech for '{audio_id}'")
        end_i = num_frames

    start_i = librosa.core.frames_to_samples(start_i, hop_length=frame_step)
    end_i = librosa.core.frames_to_samples(end_i, hop_length=frame_step)

    return start_i, end_i


def pad_sample_indices(
    start_sample_i: int, end_sample_i: int, max_sample: int, sample_rate: int, pad_seconds: float
) -> Tuple[int, int]:
    """Shift the input sample indices by pad_seconds in front and back within [0, max_sample]
       Args:
           start_sample_i: Start sample index
           end_sample_i: End sample index
           max_sample: Maximum sample index
           sample_rate: Sample rate of audio
           pad_seconds: Amount to pad/shift the indices by.

       Returns the sample indices after padding by the input amount.
    """
    pad_samples = pad_seconds * sample_rate
    start_sample_i = start_sample_i - pad_samples
    end_sample_i = end_sample_i + pad_samples

    start_sample_i = int(max(0, start_sample_i))
    end_sample_i = int(min(max_sample, end_sample_i))

    return start_sample_i, end_sample_i
