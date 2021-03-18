# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List, Optional

from nemo.collections.tts.modules.fastpitch_text.text_processing import TextProcessing


class FastPitchParser:
    def __init__(self, symbol_set='english_basic', text_cleaners=['english_cleaners']):
        self.tp = TextProcessing(symbol_set, text_cleaners)

    def __call__(self, text: str) -> Optional[List[int]]:
        return self.tp.encode_text(text, return_all=True)
