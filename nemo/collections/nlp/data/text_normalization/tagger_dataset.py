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

import os
import pickle

from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from nemo.collections.nlp.data.text_normalization import constants
from nemo.collections.nlp.data.text_normalization.utils import basic_tokenize, read_data_file
from nemo.core.classes import Dataset
from nemo.utils import logging
from nemo.utils.decorators.experimental import experimental

__all__ = ['TextNormalizationTaggerDataset']


@experimental
class TextNormalizationTaggerDataset(Dataset):
    """
    Creates dataset to use to train a DuplexTaggerModel.

    Converts from raw data to an instance that can be used by Dataloader.

    For dataset to use to do end-to-end inference, see TextNormalizationTestDataset.

    Args:
        input_file: path to the raw data file (e.g., train.tsv). For more info about the data format, refer to the `text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization.rst>`.
        tokenizer: tokenizer of the model that will be trained on the dataset
        tokenizer_name: name of the tokenizer,
        mode: should be one of the values ['tn', 'itn', 'joint'].  `tn` mode is for TN only. `itn` mode is for ITN only. `joint` is for training a system that can do both TN and ITN at the same time.
        do_basic_tokenize: a flag indicates whether to do some basic tokenization before using the tokenizer of the model
        tagger_data_augmentation (bool): a flag indicates whether to augment the dataset with additional data instances
        lang: language of the dataset
        use_cache: Enables caching to use pickle format to store and read data from,
        max_insts: Maximum number of instances (-1 means no limit)
    """

    def __init__(
        self,
        input_file: str,
        tokenizer: PreTrainedTokenizerBase,
        tokenizer_name: str,
        mode: str,
        do_basic_tokenize: bool,
        tagger_data_augmentation: bool,
        lang: str,
        use_cache: bool = False,
        max_insts: int = -1,
    ):
        assert mode in constants.MODES
        assert lang in constants.SUPPORTED_LANGS
        self.mode = mode
        self.lang = lang
        self.use_cache = use_cache
        self.max_insts = max_insts

        # Get cache path
        data_dir, filename = os.path.split(input_file)
        tokenizer_name_normalized = tokenizer_name.replace('/', '_')
        cached_data_file = os.path.join(
            data_dir, f'cached_tagger_{filename}_{tokenizer_name_normalized}_{lang}_{max_insts}.pkl'
        )

        if use_cache and os.path.exists(cached_data_file):
            logging.warning(
                f"Processing of {input_file} is skipped as caching is enabled and a cache file "
                f"{cached_data_file} already exists."
            )
            with open(cached_data_file, 'rb') as f:
                data = pickle.load(f)
                self.insts, self.tag2id, self.encodings, self.labels = data
        else:
            # Read the input raw data file, returns list of sentences parsed as list of class, w_words, s_words
            raw_insts = read_data_file(input_file)
            if max_insts >= 0:
                raw_insts = raw_insts[:max_insts]

            # Convert raw instances to TaggerDataInstance
            insts = []
            for (_, w_words, s_words) in tqdm(raw_insts):
                for inst_dir in constants.INST_DIRECTIONS:
                    if inst_dir == constants.INST_BACKWARD and mode == constants.TN_MODE:
                        continue
                    if inst_dir == constants.INST_FORWARD and mode == constants.ITN_MODE:
                        continue
                    # Create a new TaggerDataInstance
                    inst = TaggerDataInstance(w_words, s_words, inst_dir, do_basic_tokenize)
                    insts.append(inst)
                    # Data Augmentation (if enabled)
                    if tagger_data_augmentation:
                        filtered_w_words, filtered_s_words = [], []
                        for ix, (w, s) in enumerate(zip(w_words, s_words)):
                            if not s in constants.SPECIAL_WORDS:
                                filtered_w_words.append(w)
                                filtered_s_words.append(s)
                        if len(filtered_s_words) > 1:
                            inst = TaggerDataInstance(filtered_w_words, filtered_s_words, inst_dir)
                            insts.append(inst)

            self.insts = insts
            texts = [inst.input_words for inst in insts]
            tags = [inst.labels for inst in insts]

            # Tags Mapping
            self.tag2id = {tag: id for id, tag in enumerate(constants.ALL_TAG_LABELS)}

            # Finalize
            self.encodings = tokenizer(texts, is_split_into_words=True, padding=False, truncation=True)
            self.labels = self.encode_tags(tags, self.encodings)

            # Write to cache (if use_cache)
            if use_cache:
                with open(cached_data_file, 'wb') as out_file:
                    data = self.insts, self.tag2id, self.encodings, self.labels
                    pickle.dump(data, out_file, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, idx: int):
        """
        Args:
            idx: item index
        Returns:
            item: dictionary with input_ids and attention_mask as dictionary keys and the tensors at given idx as values
        """
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


class TaggerDataInstance:
    """
    This class represents a data instance in a TextNormalizationTaggerDataset.

    Args:
        w_words: List of words in a sentence in the written form
        s_words: List of words in a sentence in the spoken form
        direction: Indicates the direction of the instance (i.e., INST_BACKWARD for ITN or INST_FORWARD for TN).
        do_basic_tokenize: a flag indicates whether to do some basic tokenization before using the tokenizer of the model
    """

    def __init__(self, w_words, s_words, direction, do_basic_tokenize=False):
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
                w_word = ' '.join(basic_tokenize(w_word, self.lang))
                if not s_word in constants.SPECIAL_WORDS:
                    s_word = ' '.join(basic_tokenize(s_word, self.lang))
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
                if direction == constants.INST_BACKWARD:
                    input_words.append(s_word)
                if direction == constants.INST_FORWARD:
                    input_words.append(w_word)
                labels.append(constants.TRANSFORM_TAG)
        self.input_words = input_words
        self.labels = labels
