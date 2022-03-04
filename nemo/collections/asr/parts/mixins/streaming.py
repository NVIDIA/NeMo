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


class StreamingModuleMixin(ABC):
    @abstractmethod
    def setup_streaming_params(self, init_chunk_size=None, init_shift_size=None, chunk_size=None, shift_size=None, cache_drop_size=None):
        pass

    @abstractmethod
    def get_initial_cache_state(self, batch_size, dtype, device):
        pass

    @abstractmethod
    def streaming_forward(self, batch_size, dtype, device):
        pass
