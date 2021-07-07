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

from copy import deepcopy
from typing import List

from nltk import word_tokenize
from tqdm import tqdm

import nemo.collections.nlp.data.text_normalization.constants as constants
from nemo.collections.nlp.data.text_normalization.utils import normalize_str, read_data_file, remove_puncts
from nemo.utils.decorators.experimental import experimental

__all__ = ['TextNormalizationTestDataset']

# Test Dataset
@experimental
class TextNormalizationTestDataset:
    """
    Creates dataset to use to do end-to-end inference

    Args:
        input_file: path to the raw data file (e.g., train.tsv). For more info about the data format, refer to the `text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization.rst>`.
        mode: should be one of the values ['tn', 'itn', 'joint'].  `tn` mode is for TN only. `itn` mode is for ITN only. `joint` is for training a system that can do both TN and ITN at the same time.
        keep_puncts: whether to keep punctuations in the inputs/outputs
    """

    def __init__(self, input_file: str, mode: str, keep_puncts: bool = False):
        insts = read_data_file(input_file)

        # Build inputs and targets
        self.directions, self.inputs, self.targets = [], [], []
        for (_, w_words, s_words) in insts:
            # Extract words that are not punctuations
            processed_w_words, processed_s_words = [], []
            for w_word, s_word in zip(w_words, s_words):
                if s_word == constants.SIL_WORD:
                    if keep_puncts:
                        processed_w_words.append(w_word)
                        processed_s_words.append(w_word)
                    continue
                if s_word == constants.SELF_WORD:
                    processed_s_words.append(w_word)
                if not s_word in constants.SPECIAL_WORDS:
                    processed_s_words.append(s_word)
                processed_w_words.append(w_word)
            # Create examples
            for direction in constants.INST_DIRECTIONS:
                if direction == constants.INST_BACKWARD:
                    if mode == constants.TN_MODE:
                        continue
                    input_words = processed_s_words
                    output_words = processed_w_words
                if direction == constants.INST_FORWARD:
                    if mode == constants.ITN_MODE:
                        continue
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
    def is_same(pred: str, target: str, inst_dir: str):
        """
        Function for checking whether the predicted string can be considered
        the same as the target string

        Args:
            pred: Predicted string
            target: Target string
            inst_dir: Direction of the instance (i.e., INST_BACKWARD or INST_FORWARD).
        Return: an int value (0/1) indicating whether pred and target are the same.
        """
        if inst_dir == constants.INST_BACKWARD:
            pred = remove_puncts(pred)
            target = remove_puncts(target)
        pred = normalize_str(pred)
        target = normalize_str(target)
        return int(pred == target)

    @staticmethod
    def compute_sent_accuracy(preds: List[str], targets: List[str], inst_directions: List[str]):
        """
        Compute the sentence accuracy metric.

        Args:
            preds: List of predicted strings.
            targets: List of target strings.
            inst_directions: A list of str where each str indicates the direction of the corresponding instance (i.e., INST_BACKWARD or INST_FORWARD).
        Return: the sentence accuracy score
        """
        assert len(preds) == len(targets)
        if len(targets) == 0:
            return 'NA'
        # Sentence Accuracy
        correct_count = 0
        for inst_dir, pred, target in zip(inst_directions, preds, targets):
            correct_count += TextNormalizationTestDataset.is_same(pred, target, inst_dir)
        sent_accuracy = correct_count / len(targets)

        return sent_accuracy
