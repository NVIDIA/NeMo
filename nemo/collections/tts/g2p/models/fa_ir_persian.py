# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pathlib
from collections import defaultdict
from typing import Dict, List, Optional, Union

from nemo.collections.common.tokenizers.text_to_speech.ipa_lexicon import (
    get_grapheme_character_set,
    get_ipa_punctuation_list,
)
from nemo.collections.tts.g2p.models.base import BaseG2p
from nemo.collections.tts.g2p.models.persian.phonemizer import PersianPhonemizer
from nemo.collections.tts.g2p.utils import set_grapheme_case
from nemo.utils import logging


class PersianG2p(BaseG2p):
    def __init__(
        self,
        phoneme_dict=None,
        ignore_ambiguous_words=True,
        heteronyms=None,
        encoding='utf-8',
        phoneme_probability: Optional[float] = None,
        mapping_file: Optional[str] = None,
    ):
        """
        Persian G2P module.
        Args:
            to be defined.
        """
        assert phoneme_dict is not None, "Please set the phoneme_dict path."
        self.phoner = PersianPhonemizer(dictionary_path=phoneme_dict)

    def __call__(self, text):
        phoneme_seq = self.phoner.phonemize(text)
        return phoneme_seq
