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

import nemo.collections.nlp.data.text_normalization.constants as constants

from tqdm import tqdm
from copy import deepcopy
from nltk import word_tokenize
from typing import List

from nemo.collections.nlp.data.text_normalization.utils import read_data_file, normalize_str

__all__ = ['TextNormalizationTestDataset']

# Test Dataset
class TextNormalizationTestDataset:
    def __init__(self, input_file: str, mode: str):
        insts = read_data_file(input_file)

        # Build inputs and targets
        self.directions, self.inputs, self.targets = [], [], []
        for (_, w_words, s_words) in insts:
            # Extract words that are not punctuations
            processed_w_words, processed_s_words = [], []
            for w_word, s_word in zip(w_words, s_words):
                if s_word == constants.SIL_WORD: continue
                if s_word == constants.SELF_WORD: processed_s_words.append(w_word)
                if not s_word in constants.SPECIAL_WORDS: processed_s_words.append(s_word)
                processed_w_words.append(w_word)
            # Create examples
            for direction in constants.INST_DIRECTIONS:
                if direction == constants.INST_BACKWARD:
                    if mode == constants.TN_MODE: continue
                    input_words = processed_s_words
                    output_words = processed_w_words
                if direction == constants.INST_FORWARD:
                    if mode == constants.ITN_MODE: continue
                    input_words = w_words
                    output_words = processed_s_words
                # Basic tokenization
                input_words = word_tokenize(' '.join(input_words))
                output_words = word_tokenize(' '.join(output_words))
                # Update self.directions, self.inputs, self.targets
                self.directions.append(direction)
                self.inputs.append(' '.join(input_words))
                self.targets.append(' '.join(output_words))
        self.examples = list(zip(self.directions, self.inputs, self.targets))

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.inputs)

    @staticmethod
    def compute_sent_accuracy(preds: List[str], targets: List[str]):
        assert(len(preds) == len(targets))
        if len(targets) == 0: return 'NA'
        # Sentence Accuracy
        correct_count = 0
        for pred, target in zip(preds, targets):
            pred = normalize_str(pred)
            target = normalize_str(target)
            correct_count += int(pred == target)
        sent_accuracy = correct_count / len(targets)

        return sent_accuracy
