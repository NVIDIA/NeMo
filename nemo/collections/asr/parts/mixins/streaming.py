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

import torch


class StreamingEncoderMixin(ABC):
    @abstractmethod
    def setup_streaming_params(
        self, init_chunk_size=None, init_shift_size=None, chunk_size=None, shift_size=None, cache_drop_size=None
    ):
        pass

    @abstractmethod
    def get_initial_cache_state(self, batch_size, dtype, device):
        pass

    @abstractmethod
    def streaming_forward(self, batch_size, dtype, device):
        pass

    @staticmethod
    def to_numpy(tensor):
        if tensor is None:
            return None
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    def stream_step(
        self,
        processed_signal,
        processed_signal_length=None,
        cache_last_channel=None,
        cache_last_time=None,
        valid_out_len=None,
        drop_extra_pre_encoded=None,
        onnx_model=None
    ):
        if self.streaming_cfg is None:
            self.setup_streaming_params()
        if drop_extra_pre_encoded is not None:
            prev_drop_extra_pre_encoded = self.streaming_cfg.drop_extra_pre_encoded
            self.streaming_cfg.drop_extra_pre_encoded = drop_extra_pre_encoded
        else:
            prev_drop_extra_pre_encoded = None

        if processed_signal_length is None:
            processed_signal_length = processed_signal.new_full(processed_signal.size(0), processed_signal.size(-1))

        if onnx_model is None:
            encoder_output = self(
                audio_signal=processed_signal,
                length=processed_signal_length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
            )
        else:
            ort_inputs = {
                onnx_model.get_inputs()[0].name: self.to_numpy(processed_signal),
                onnx_model.get_inputs()[1].name: self.to_numpy(processed_signal_length),
                onnx_model.get_inputs()[2].name: self.to_numpy(cache_last_channel),
                onnx_model.get_inputs()[3].name: self.to_numpy(cache_last_time),
            }
            encoder_output = onnx_model.run(None, ort_inputs)
            for idx, t in enumerate(encoder_output):
                encoder_output[idx] = torch.tensor(t).to(processed_signal.device)

        if len(encoder_output) == 2:
            encoded, encoded_len = encoder_output
            cache_last_channel_next = cache_last_time_next = None
        else:
            encoded, encoded_len, cache_last_channel_next, cache_last_time_next = encoder_output

        if valid_out_len is not None:
            encoded = encoded[:, :, :valid_out_len]
            encoded_len = torch.clamp(encoded_len, max=valid_out_len)

        return encoded, encoded_len, cache_last_channel_next, cache_last_time_next, prev_drop_extra_pre_encoded
