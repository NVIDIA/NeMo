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


from typing import List, Optional, Tuple

from torch import Tensor

from nemo.collections.asr.inference.asr.asr_inference import ASRInference


class CacheAwareContext:
    def __init__(
        self,
        cache_last_channel: Optional[Tensor] = None,
        cache_last_time: Optional[Tensor] = None,
        cache_last_channel_len: Optional[Tensor] = None,
    ):
        self.cache_last_channel = cache_last_channel
        self.cache_last_time = cache_last_time
        self.cache_last_channel_len = cache_last_channel_len

    def set_cache_last_channel(self, cache_last_channel: Tensor) -> None:
        self.cache_last_channel = cache_last_channel

    def set_cache_last_time(self, cache_last_time: Tensor) -> None:
        self.cache_last_time = cache_last_time

    def set_cache_last_channel_len(self, cache_last_channel_len: Tensor) -> None:
        self.cache_last_channel_len = cache_last_channel_len


class CacheAwareASRInference(ASRInference):

    def get_input_features(self) -> int:
        """Returns:
        (int) number of channels in the input features.
        """
        return self.asr_model.encoder._feat_in

    def get_sampling_frames(self) -> List[int] | int | None:
        """
        It is used for checking to make sure the audio chunk has enough frames to produce at least one output after downsampling.
        Returns:
            (List[int] | int | None) sampling frames for the encoder.
        """
        self.sampling_frames = None
        if hasattr(self.asr_model.encoder, "pre_encode") and hasattr(
            self.asr_model.encoder.pre_encode, "get_sampling_frames"
        ):
            self.sampling_frames = self.asr_model.encoder.pre_encode.get_sampling_frames()
        return self.sampling_frames

    def get_initial_cache_state(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns:
        (Tuple[Tensor, Tensor, Tensor]) the initial cache state of the encoder.
        """
        return self.asr_model.encoder.get_initial_cache_state(batch_size=batch_size)

    def get_drop_extra_pre_encoded(self) -> int:
        """Returns:
        (int) drop_extra_pre_encoded.
        """
        return self.asr_model.encoder.streaming_cfg.drop_extra_pre_encoded

    def get_chunk_size(self) -> List[int] | int:
        """Returns:
        (List[int] | int) the chunk size.
        """
        return self.asr_model.encoder.streaming_cfg.chunk_size

    def get_shift_size(self) -> List[int] | int:
        """Returns:
        (List[int] | int) the shift size.
        """
        return self.asr_model.encoder.streaming_cfg.shift_size

    def get_pre_encode_cache_size(self) -> List[int] | int:
        """Returns:
        (List[int] | int) the pre_encode cache size.
        """
        return self.asr_model.encoder.streaming_cfg.pre_encode_cache_size

    def get_subsampling_factor(self) -> int:
        """Returns:
        (int) subsampling factor for the ASR encoder model.
        """
        return self.asr_model.encoder.subsampling_factor

    def get_att_context_size(self) -> List:
        """Returns:
        (List) the attention context size.
        """
        return self.asr_model.encoder.att_context_size.copy()

    def set_default_att_context_size(self, att_context_size: List) -> None:
        """
        Set the default attention context size for the encoder.
        The list of the supported look-ahead: [[70, 13], [70, 6], [70, 1], [70, 0]]
        Args:
            att_context_size: (List) the attention context size.
        """
        if hasattr(self.asr_model.encoder, "set_default_att_context_size"):
            self.asr_model.encoder.set_default_att_context_size(att_context_size=att_context_size)
        else:
            raise ValueError("Model does not support multiple lookaheads.")

    def setup_streaming_params(self, chunk_size: int, shift_size: int) -> None:
        """
        Setup the streaming parameters (chunk_size, shift_size) for the encoder.
        Args:
            chunk_size: (int) the chunk size.
            shift_size: (int) the shift size.
        """
        self.asr_model.encoder.setup_streaming_params(chunk_size=chunk_size, shift_size=shift_size)

    def stream_step(self) -> None:
        """Executes a single streaming step."""
        raise NotImplementedError(
            "`stream_step` method is not implemented. It is required for cache-aware transcribers."
        )
