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


from enum import Enum, auto


class ASRDecodingType(Enum):
    CTC = auto()
    RNNT = auto()
    AED = auto()

    @classmethod
    def from_str(cls, type_name: str) -> 'ASRDecodingType':
        """Convert a string to an ASRDecodingType enum value."""
        if type_name.lower() == "ctc":
            return ASRDecodingType.CTC
        elif type_name.lower() == "rnnt":
            return ASRDecodingType.RNNT
        elif type_name.lower() == "aed":
            return ASRDecodingType.AED

        choices = [choice.name for choice in cls]
        raise ValueError(f"Invalid ASR decoding type `{type_name}`: Need to be one of {choices}")


class RecognizerType(Enum):
    BUFFERED_STREAMING = auto()
    CACHE_AWARE_STREAMING = auto()

    @classmethod
    def from_str(cls, type_name: str) -> 'RecognizerType':
        """Convert a string to a RecognizerType enum value."""
        if type_name.lower() == "buffered_streaming":
            return RecognizerType.BUFFERED_STREAMING
        elif type_name.lower() == "cache_aware_streaming":
            return RecognizerType.CACHE_AWARE_STREAMING

        choices = [choice.name for choice in cls]
        raise ValueError(f"Invalid recognizer type `{type_name}`: Need to be one of {choices}")

