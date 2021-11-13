# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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


class TTSDataType:
    name = None


class WithLens(TTSDataType):
    """Represent that this TTSDataType returns lengths for data"""


class Audio(WithLens):
    name = "audio"


class Text(WithLens):
    name = "text"


class LogMel(WithLens):
    name = "log_mel"


class Durations(TTSDataType):
    name = "durations"


class DurationPrior(TTSDataType):
    name = "duration_prior"


class Pitch(WithLens):
    name = "pitch"


class Energy(WithLens):
    name = "energy"


class LMTokens:
    name = "lm_tokens"


MAIN_DATA_TYPES = [Audio, Text]
VALID_SUPPLEMENTARY_DATA_TYPES = [LogMel, Durations, DurationPrior, Pitch, Energy, LMTokens]
DATA_STR2DATA_CLASS = {d.name: d for d in MAIN_DATA_TYPES + VALID_SUPPLEMENTARY_DATA_TYPES}
