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
import random
from collections import OrderedDict
from typing import List

from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from nemo.collections.nlp.data.text_normalization import constants
from nemo.collections.nlp.data.text_normalization.utils import basic_tokenize, read_data_file
from nemo.core.classes import Dataset
from nemo.utils import logging
from nemo.utils.decorators.experimental import experimental

__all__ = ['TextNormalizationDecoderDataset']


@experimental
class TextNormalizationDecoderDataset(Dataset):
    """
    Creates dataset to use to train a DuplexDecoderModel.

    Converts from raw data to an instance that can be used by Dataloader.

    For dataset to use to do end-to-end inference, see TextNormalizationTestDataset.

    Args:
        input_file: path to the raw data file (e.g., train.tsv). For more info about the data format, refer to the `text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization.rst>`.
        tokenizer: tokenizer of the model that will be trained on the dataset
        tokenizer_name: name of the tokenizer,
        mode: should be one of the values ['tn', 'itn', 'joint'].  `tn` mode is for TN only. `itn` mode is for ITN only. `joint` is for training a system that can do both TN and ITN at the same time.
        max_len: maximum length of sequence in tokens. The code will discard any training instance whose input or output is longer than the specified max_len.
        decoder_data_augmentation (bool): a flag indicates whether to augment the dataset with additional data instances that may help the decoder become more robust against the tagger's errors. Refer to the doc for more info.
        lang: language of the dataset
        do_basic_tokenize: a flag indicates whether to do some basic tokenization for the inputs
        use_cache: Enables caching to use pickle format to store and read data from
        max_insts: Maximum number of instances (-1 means no limit)
    """

    def __init__(
        self,
        input_file: str,
        tokenizer: PreTrainedTokenizerBase,
        tokenizer_name: str,
        mode: str,
        max_len: int,
        decoder_data_augmentation: bool,
        lang: str,
        do_basic_tokenize: bool,
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
            data_dir, f'cached_decoder_{filename}_{tokenizer_name_normalized}_{lang}_{max_insts}_{mode}.pkl'
        )

        if use_cache and os.path.exists(cached_data_file):
            logging.warning(
                f"Processing of {input_file} is skipped as caching is enabled and a cache file "
                f"{cached_data_file} already exists."
            )
            with open(cached_data_file, 'rb') as f:
                data = pickle.load(f)
                self.insts, self.inputs, self.examples, self.tn_count, self.itn_count, self.label_ids_semiotic = data
        else:
            raw_insts = read_data_file(fp=input_file, max_insts=max_insts)
            all_semiotic_classes = set([])
            # Convert raw instances to TaggerDataInstance
            insts = []
            for (classes, w_words, s_words) in tqdm(raw_insts):
                for ix, (_class, w_word, s_word) in enumerate(zip(classes, w_words, s_words)):
                    all_semiotic_classes.update([_class])
                    if s_word in constants.SPECIAL_WORDS:
                        continue
                    for inst_dir in constants.INST_DIRECTIONS:
                        if inst_dir == constants.INST_BACKWARD and mode == constants.TN_MODE:
                            continue
                        if inst_dir == constants.INST_FORWARD and mode == constants.ITN_MODE:
                            continue
                        # Create a DecoderDataInstance
                        inst = DecoderDataInstance(
                            w_words,
                            s_words,
                            inst_dir,
                            start_idx=ix,
                            end_idx=ix + 1,
                            lang=self.lang,
                            semiotic_class=_class,
                            do_basic_tokenize=do_basic_tokenize,
                        )
                        insts.append(inst)

                        if decoder_data_augmentation:
                            noise_left = random.randint(1, 2)
                            noise_right = random.randint(1, 2)
                            inst = DecoderDataInstance(
                                w_words,
                                s_words,
                                inst_dir,
                                start_idx=ix - noise_left,
                                end_idx=ix + 1 + noise_right,
                                semiotic_class=_class,
                                lang=self.lang,
                                do_basic_tokenize=do_basic_tokenize,
                            )
                            insts.append(inst)

            all_semiotic_classes = list(all_semiotic_classes)
            all_semiotic_classes.sort()
            self.label_ids_semiotic = OrderedDict({l: idx for idx, l in enumerate(all_semiotic_classes)})
            logging.info(f'Label_ids: {self.label_ids_semiotic}')

            # save labels list from the training file to the input_file to the file
            dir_name, file_name = os.path.split(input_file)
            if 'train' in file_name:
                with open(os.path.join(dir_name, f"label_ids_{file_name}"), 'w') as f:
                    f.write('\n'.join(self.label_ids_semiotic.keys()))

            self.insts = insts
            inputs = [inst.input_str.strip() for inst in insts]
            inputs_center = [inst.input_center_str.strip() for inst in insts]
            targets = [inst.output_str.strip() for inst in insts]
            classes = [self.label_ids_semiotic[inst.semiotic_class] for inst in insts]
            directions = [constants.DIRECTIONS_TO_ID[inst.direction] for inst in insts]

            # Tokenization
            self.inputs, self.examples, _inputs_center = [], [], []
            self.tn_count, self.itn_count, long_examples_filtered = 0, 0, 0
            input_max_len, target_max_len = 0, 0
            for idx in range(len(inputs)):
                # Input
                _input = tokenizer([inputs[idx]])
                input_len = len(_input['input_ids'][0])
                if input_len > max_len:
                    long_examples_filtered += 1
                    continue

                # Target
                _target = tokenizer([targets[idx]])
                target_len = len(_target['input_ids'][0])
                if target_len > max_len:
                    long_examples_filtered += 1
                    continue

                # Update
                self.inputs.append(inputs[idx])
                _input['labels'] = _target['input_ids']
                _input['semiotic_class_id'] = [classes[idx]]
                _input['direction'] = [directions[idx]]
                _inputs_center.append(inputs_center[idx])

                self.examples.append(_input)
                if inputs[idx].startswith(constants.TN_PREFIX):
                    self.tn_count += 1
                if inputs[idx].startswith(constants.ITN_PREFIX):
                    self.itn_count += 1
                input_max_len = max(input_max_len, input_len)
                target_max_len = max(target_max_len, target_len)
            print(f'long_examples_filtered: {long_examples_filtered}')
            print(f'input_max_len: {input_max_len} | target_max_len: {target_max_len}')

            # we need to pad input_center, so we first collect all values, and then batch_tokenize with padding
            _input_centers = tokenizer(_inputs_center, padding=True)

            for idx in range(len(self.examples)):
                self.examples[idx]['input_center'] = [_input_centers['input_ids'][idx]]

            # Write to cache (if use_cache)
            if use_cache:
                with open(cached_data_file, 'wb') as out_file:
                    data = (
                        self.insts,
                        self.inputs,
                        self.examples,
                        self.tn_count,
                        self.itn_count,
                        self.label_ids_semiotic,
                    )
                    pickle.dump(data, out_file, protocol=pickle.HIGHEST_PROTOCOL)

    def __getitem__(self, idx):
        """
        Returns:
            'input_ids': input ids
            'attention_mask': attention mask
            'labels': ground truth labels
            'semiotic_class_id': id of the semiotic class of the example
            'direction_id': id of the TN/ITN tast (see constants for the values)
            'inputs_center': ids of input center (only semiotic span, no special tokens and context)
        """
        example = self.examples[idx]
        item = {key: val[0] for key, val in example.items()}
        return item

    def __len__(self):
        return len(self.examples)


class DecoderDataInstance:
    """
    This class represents a data instance in a TextNormalizationDecoderDataset.

    Intuitively, each data instance can be thought as having the following form:
        Input:  <Left Context of Input> <Input Span> <Right Context of Input>
        Output: <Output Span>
    where the context size is determined by the constant DECODE_CTX_SIZE.

    Args:
        w_words: List of words in the written form
        s_words: List of words in the spoken form
        inst_dir: Indicates the direction of the instance (i.e., INST_BACKWARD for ITN or INST_FORWARD for TN).
        start_idx: The starting index of the input span in the original input text
        end_idx: The ending index of the input span (exclusively)
        lang: Language of the instance
        semiotic_class: The semiotic class of the input span (can be set to None if not available)
        do_basic_tokenize: a flag indicates whether to do some basic tokenization for the inputs
    """

    def __init__(
        self,
        w_words: List[str],
        s_words: List[str],
        inst_dir: str,
        start_idx: int,
        end_idx: int,
        lang: str,
        semiotic_class: str = None,
        do_basic_tokenize: bool = False,
    ):
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, len(w_words))
        ctx_size = constants.DECODE_CTX_SIZE
        extra_id_0 = constants.EXTRA_ID_0
        extra_id_1 = constants.EXTRA_ID_1

        # Extract center words
        c_w_words = w_words[start_idx:end_idx]
        c_s_words = s_words[start_idx:end_idx]

        # Extract context
        w_left = w_words[max(0, start_idx - ctx_size) : start_idx]
        w_right = w_words[end_idx : end_idx + ctx_size]
        s_left = s_words[max(0, start_idx - ctx_size) : start_idx]
        s_right = s_words[end_idx : end_idx + ctx_size]

        # Process sil words and self words
        for jx in range(len(s_left)):
            if s_left[jx] == constants.SIL_WORD:
                s_left[jx] = ''
            if s_left[jx] == constants.SELF_WORD:
                s_left[jx] = w_left[jx]
        for jx in range(len(s_right)):
            if s_right[jx] == constants.SIL_WORD:
                s_right[jx] = ''
            if s_right[jx] == constants.SELF_WORD:
                s_right[jx] = w_right[jx]
        for jx in range(len(c_s_words)):
            if c_s_words[jx] == constants.SIL_WORD:
                c_s_words[jx] = ''
                if inst_dir == constants.INST_BACKWARD:
                    c_w_words[jx] = ''
            if c_s_words[jx] == constants.SELF_WORD:
                c_s_words[jx] = c_w_words[jx]

        # Extract input_words and output_words
        if do_basic_tokenize:
            c_w_words = basic_tokenize(' '.join(c_w_words), lang)
            c_s_words = basic_tokenize(' '.join(c_s_words), lang)
        w_input = w_left + [extra_id_0] + c_w_words + [extra_id_1] + w_right
        s_input = s_left + [extra_id_0] + c_s_words + [extra_id_1] + s_right
        if inst_dir == constants.INST_BACKWARD:
            input_center_words = c_s_words
            input_words = [constants.ITN_PREFIX] + s_input
            output_words = c_w_words
        if inst_dir == constants.INST_FORWARD:
            input_center_words = c_w_words
            input_words = [constants.TN_PREFIX] + w_input
            output_words = c_s_words
        # Finalize
        self.input_str = ' '.join(input_words)
        self.input_center_str = ' '.join(input_center_words)
        self.output_str = ' '.join(output_words)
        self.direction = inst_dir
        self.semiotic_class = semiotic_class
