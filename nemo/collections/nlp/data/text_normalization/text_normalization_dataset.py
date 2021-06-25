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

from tqdm import tqdm
from copy import deepcopy
from nltk import word_tokenize
from nemo.core.classes import Dataset
from typing import Dict, List, Optional
from transformers import PreTrainedTokenizerBase
from nemo.collections.nlp.data.text_normalization.constants import *

__all__ = ['TextNormalizationTaggerDataset', 'TextNormalizationTestDataset']


# Tagger Dataset
class TaggerDataInstance:
    def __init__(self, w_words, s_words, do_basic_tokenize=False):
        # Build input_words and labels
        input_words, labels = [], []
        # Task Prefix
        input_words.append(TN_PREFIX)
        labels.append(TASK_TAG)
        # Main Content
        for w_word, s_word in zip(w_words, s_words):
            if do_basic_tokenize:
                w_word = ' '.join(word_tokenize(w_word))
            input_words.append(w_word)
            if s_word == SELF_WORD: labels.append(SAME_TAG)
            elif s_word == SIL_WORD: labels.append(PUNCT_TAG)
            else: labels.append(TRANSFORM_TAG)
        self.input_words = input_words
        self.labels = labels

class TextNormalizationTaggerDataset(Dataset):
    def __init__(
        self,
        input_file: str,
        tokenizer: PreTrainedTokenizerBase,
        do_basic_tokenize: bool
    ):
        raw_insts = read_data_file(input_file)

        # Convert raw instances to TaggerDataInstance
        insts, texts, tags = [], [], []
        for (_, w_words, s_words) in tqdm(raw_insts):
            inst = TaggerDataInstance(w_words, s_words, do_basic_tokenize)
            insts.append(inst)
            texts.append(inst.input_words)
            tags.append(inst.labels)
        self.insts = insts

        # Tags Mapping
        self.tag2id = {tag: id for id, tag in enumerate(ALL_TAG_LABELS)}

        # Finalize
        self.encodings = tokenizer(texts, is_split_into_words=True,
                                   padding=False, truncation=True)
        self.labels = self.encode_tags(tags, self.encodings)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

    def encode_tags(self, tags, encodings):
        encoded_labels = []
        for i, label in enumerate(tags):
            word_ids = encodings.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label
                # to -100 (LABEL_PAD_TOKEN_ID) so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(LABEL_PAD_TOKEN_ID)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_id = self.tag2id[B_PREFIX + label[word_idx]]
                    label_ids.append(label_id)
                # We set the label for the other tokens in a word
                else:
                    label_id = self.tag2id[I_PREFIX + label[word_idx]]
                    label_ids.append(label_id)
                previous_word_idx = word_idx

            encoded_labels.append(label_ids)

        return encoded_labels

# Test Dataset
class TextNormalizationTestDataset:
    def __init__(self, input_file: str):
        insts = read_data_file(input_file)

        # Build inputs and targets
        self.inputs, self.targets = [], []
        for (_, w_words, s_words) in insts:
            processed_s_words = []
            for w_word, s_word in zip(w_words, s_words):
                if s_word == SIL_WORD: continue
                if s_word == SELF_WORD: processed_s_words.append(w_word)
                if not s_word in SPECIAL_WORDS: processed_s_words.append(s_word)
            input_words = word_tokenize(' '.join(w_words))
            output_words = word_tokenize(' '.join(processed_s_words))

            self.inputs.append(' '.join(input_words))
            self.targets.append(' '.join(output_words))
            self.examples = list(zip(self.inputs, self.targets))

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.inputs)

    @staticmethod
    def compute_sent_accuracy(preds: List[str], targets: List[str]):
        assert(len(preds) == len(targets))
        # Sentence Accuracy
        correct_count = 0
        for pred, target in zip(preds, targets):
            pred = normalize_str(pred)
            target = normalize_str(target)
            correct_count += int(pred == target)
        sent_accuracy = correct_count / len(targets)

        return sent_accuracy

# Helper Functions
def read_data_file(fp):
    insts, w_words, s_words, classes = [], [], [], []
    # Read input file
    with open(fp, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            es = [e.strip() for e in line.strip().split('\t')]
            if es[0] == '<eos>':
                inst = (deepcopy(classes), deepcopy(w_words), deepcopy(s_words))
                insts.append(inst)
                # Reset
                w_words, s_words, classes = [], [], []
            else:
                classes.append(es[0])
                w_words.append(es[1])
                s_words.append(es[2])
    return insts

def normalize_str(input_str):
    input_str = ' '.join(word_tokenize(input_str.strip().lower()))
    input_str = input_str.replace('  ', ' ')
    return input_str
