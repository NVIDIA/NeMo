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
from nltk import word_tokenize
from nemo.core.classes import Dataset
from transformers import PreTrainedTokenizerBase

from nemo.collections.nlp.data.text_normalization.utils import read_data_file

__all__ = ['TextNormalizationTaggerDataset']

# Tagger Dataset
class TaggerDataInstance:
    def __init__(self, w_words, s_words, direction, do_basic_tokenize = False):
        # Build input_words and labels
        input_words, labels = [], []
        # Task Prefix
        if direction == constants.INST_BACKWARD:
            input_words.append(constants.ITN_PREFIX)
        if direction == constants.INST_FORWARD:
            input_words.append(constants.TN_PREFIX)
        labels.append(constants.TASK_TAG)
        # Main Content
        for w_word, s_word in zip(w_words, s_words):
            # Basic tokenization (if enabled)
            if do_basic_tokenize:
                w_word = ' '.join(word_tokenize(w_word))
                if not s_word in constants.SPECIAL_WORDS:
                    s_word = ' '.join(word_tokenize(s_word))
            # Update input_words and labels
            if s_word == constants.SIL_WORD and direction == constants.INST_BACKWARD:
                continue
            if s_word == constants.SELF_WORD:
                input_words.append(w_word)
                labels.append(constants.SAME_TAG)
            elif s_word == constants.SIL_WORD:
                input_words.append(w_word)
                labels.append(constants.PUNCT_TAG)
            else:
                if direction == constants.INST_BACKWARD: input_words.append(s_word)
                if direction == constants.INST_FORWARD: input_words.append(w_word)
                labels.append(constants.TRANSFORM_TAG)
        self.input_words = input_words
        self.labels = labels

class TextNormalizationTaggerDataset(Dataset):
    def __init__(
        self,
        input_file: str,
        tokenizer: PreTrainedTokenizerBase,
        mode: str,
        do_basic_tokenize: bool,
    ):
        assert(mode in constants.MODES)
        self.mode = mode
        raw_insts = read_data_file(input_file)

        # Convert raw instances to TaggerDataInstance
        insts, texts, tags = [], [], []
        for (_, w_words, s_words) in tqdm(raw_insts):
            for inst_dir in constants.INST_DIRECTIONS:
                if inst_dir == constants.INST_BACKWARD and mode == constants.TN_MODE: continue
                if inst_dir == constants.INST_FORWARD and mode == constants.ITN_MODE: continue
                # Create a new TaggerDataInstance
                inst = TaggerDataInstance(w_words, s_words, inst_dir, do_basic_tokenize)
                insts.append(inst)
                texts.append(inst.input_words)
                tags.append(inst.labels)
        self.insts = insts

        # Tags Mapping
        self.tag2id = {tag: id for id, tag in enumerate(constants.ALL_TAG_LABELS)}

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
                    label_ids.append(constants.LABEL_PAD_TOKEN_ID)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_id = self.tag2id[constants.B_PREFIX + label[word_idx]]
                    label_ids.append(label_id)
                # We set the label for the other tokens in a word
                else:
                    label_id = self.tag2id[constants.I_PREFIX + label[word_idx]]
                    label_ids.append(label_id)
                previous_word_idx = word_idx

            encoded_labels.append(label_ids)

        return encoded_labels
