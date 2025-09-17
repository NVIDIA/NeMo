# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import numbers
from typing import List, Tuple

import torch
from omegaconf import DictConfig

from nemo.collections.asr.inference.stream.buffering.audio_bufferer import AudioBufferer
from nemo.collections.asr.inference.stream.framing.request import Frame
from nemo.collections.audio.models import AudioToAudioModel


class CacheSTFTFeatureBufferer:

    def __init__(
        self,
        sample_rate: int,
        chunk_size_in_secs: float,
        window_size: int,
        model_cfg: DictConfig,
        device: torch.device,
        fill_value: complex | float = 0.0,
        window: torch.Tensor | None = None,
    ):

        self.n_feat = model_cfg.estimator.feat_in
        self.sample_rate = sample_rate
        self.chunk_size_in_secs = chunk_size_in_secs
        self.window_size = window_size
        self.hop_length = model_cfg.encoder.hop_length
        self.device = device

        # STFT feature buffer stores complex values. Coerce provided fill_value to complex.
        if isinstance(fill_value, numbers.Number):
            self.ZERO_LEVEL_SPEC_DB_VAL = complex(fill_value)
        else:
            self.ZERO_LEVEL_SPEC_DB_VAL = complex(0.0, 0.0)

        # Derive how many STFT frames we should emit per update step from time-based configuration
        # frames_per_step = round(chunk_size_in_secs / (hop_length / sample_rate))
        self.timestep_duration = self.hop_length / self.sample_rate
        self.frames_per_step = max(1, int(round(self.chunk_size_in_secs / self.timestep_duration)))

        # Decide buffer sizing policy using the derived frames_per_step
        # Keep only what is required to compute `frames_per_step` consecutive STFT frames
        # with stride == hop_length and center=False: window + (k-1)*hop
        self.buffer_size = self.window_size + (self.frames_per_step - 1) * self.hop_length

        self.buffer_size_in_secs = self.buffer_size / self.sample_rate
        self.chunk_size = int(self.chunk_size_in_secs * self.sample_rate)

        self.sample_buffer = AudioBufferer(sample_rate, self.buffer_size_in_secs)

        # Maintain only `frames_per_step` frames in the feature buffer and emit the same per update
        self.feature_buffer_len = self.frames_per_step
        self.feature_chunk_len = self.frames_per_step

        self.feature_buffer = torch.full(
            [1, self.n_feat, self.feature_buffer_len],
            self.ZERO_LEVEL_SPEC_DB_VAL,
            dtype=torch.complex64,
            device=self.device,
        )

        self.current_buffer_size = 0

        self.preprocessor = AudioToAudioModel.from_config_dict(model_cfg.encoder)
        self.preprocessor.to(self.device)
        self.preprocessor.center = False

        if window is not None:
            self.preprocessor.window = window.to(self.device)

    def reset(self) -> None:
        """
        Reset the buffer to zero
        """
        self.sample_buffer.reset()
        self.feature_buffer.fill_(self.ZERO_LEVEL_SPEC_DB_VAL)
        self.current_buffer_size = 0

    def _update_feature_buffer(self, feat_chunk: torch.Tensor) -> None:
        """
        Add an extracted feature to `feature_buffer`
        """
        self.feature_buffer[:, :, : -self.feature_chunk_len] = self.feature_buffer[
            :, :, self.feature_chunk_len :
        ].clone()
        self.feature_buffer[:, :, -self.feature_chunk_len :] = feat_chunk.clone()

    def preprocess(self, audio_signal: torch.Tensor) -> torch.Tensor:
        """
        Preprocess the audio signal using the preprocessor
        Args:
            audio_signal (torch.Tensor): audio signal
        Returns:
            torch.Tensor: preprocessed features
        """
        audio_signal = audio_signal.unsqueeze_(0).unsqueeze_(0).to(self.device)
        audio_signal_len = torch.tensor([audio_signal.shape[2]], device=self.device)
        features, _ = self.preprocessor(
            input=audio_signal,
            input_length=audio_signal_len,
        )
        features = features.squeeze(0)
        return features

    def update(self, frame: Frame) -> None:
        """
        Update the sample anf feature buffers with the new frame
        Args:
            frame (Frame): frame to update the buffer with
        """

        # Update the sample buffer with the new frame
        self.sample_buffer.update(frame)
        self.current_buffer_size += frame.size

        # Take exactly what is needed to produce `feature_chunk_len` frames
        required_samples = self.window_size + (self.feature_chunk_len - 1) * self.hop_length
        samples = self.sample_buffer.sample_buffer[-required_samples:]

        # Get the stft spectrogram
        features = self.preprocess(samples)

        # Update the feature buffer with the new features
        self._update_feature_buffer(features[:, :, -self.feature_chunk_len :])

    def get_buffer(self) -> torch.Tensor:
        """
        Get the current sample buffer
        Returns:
            torch.Tensor: current state of the buffer
        """
        return self.sample_buffer.get_buffer()

    def get_feature_buffer(self) -> torch.Tensor:
        """
        Get the current feature buffer
        Returns:
            torch.Tensor: current state of the feature buffer
        """
        return self.feature_buffer.clone()


class BatchedCacheSTFTFeatureBufferer:

    def __init__(
        self,
        sample_rate: int,
        chunk_size_in_secs: float,
        window_size: int,
        model_cfg: DictConfig,
        device: torch.device,
        window: torch.Tensor | None = None,
    ):

        self.hop_length = model_cfg.encoder.hop_length

        self.window = window
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.chunk_size_in_secs = chunk_size_in_secs

        self.device = device
        self.model_cfg = model_cfg

        self.bufferers = {}

    def reset(self) -> None:
        """
        Reset bufferers
        """
        self.bufferers = {}

    def rm_bufferer(self, stream_id: int) -> None:
        """
        Remove bufferer for the given stream id
        Args:
            stream_id (int): stream id
        """
        self.bufferers.pop(stream_id, None)

    def update(self, frames: List[Frame]) -> Tuple[List, List]:
        """
        Update the feature bufferers with the new frames.
        Frames can come from different streams (audios), so we need to maintain a bufferer for each stream.
        Args:
            frames (List[Frame]): list of frames
        Returns:
            list of pre-processed buffered and unbuffered frames
        """
        buffered_frames = []
        ready_frames = []
        for idx, frame in enumerate(frames):
            bufferer = self.bufferers.get(frame.stream_id, None)

            if bufferer is None:
                bufferer = CacheSTFTFeatureBufferer(
                    sample_rate=self.sample_rate,
                    chunk_size_in_secs=self.chunk_size_in_secs,
                    window_size=self.window_size,
                    model_cfg=self.model_cfg,
                    device=self.device,
                    window=self.window,
                )
                self.bufferers[frame.stream_id] = bufferer

            bufferer.update(frame)
            buffered_frames.append((idx, bufferer.get_feature_buffer()))

            # Ready when enough samples to compute exactly K frames are present in the sample buffer
            required_samples = bufferer.window_size + (bufferer.feature_chunk_len - 1) * bufferer.hop_length
            ready_frames.append(bufferer.current_buffer_size >= required_samples)

            if frame.is_last:
                self.rm_bufferer(frame.stream_id)

        return buffered_frames, ready_frames
