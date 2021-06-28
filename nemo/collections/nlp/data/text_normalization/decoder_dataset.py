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

import random

from tqdm import tqdm
from nltk import word_tokenize
from transformers import PreTrainedTokenizerBase
from nemo.collections.nlp.data.text_normalization.constants import *
from nemo.collections.nlp.data.text_normalization.utils import *

__all__ = ['TextNormalizationDecoderDataset']

# Decoder Dataset
class DecoderDataInstance:
    def __init__(self, w_words, s_words, start_idx, end_idx, semiotic_class=None):
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, len(w_words))

        # Extract center words
        c_w_words = w_words[start_idx:end_idx]
        c_s_words = s_words[start_idx:end_idx]

        # Extract context
        w_left  = w_words[max(0,start_idx-DECODE_CTX_SIZE):start_idx]
        w_right = w_words[end_idx:end_idx+DECODE_CTX_SIZE]
        s_left  = s_words[max(0,start_idx-DECODE_CTX_SIZE):start_idx]
        s_right = s_words[end_idx:end_idx+DECODE_CTX_SIZE]

        # Process sil words and self words
        for jx in range(len(s_left)):
            if s_left[jx] == SIL_WORD: s_left[jx] = ''
            if s_left[jx] == SELF_WORD: s_left[jx] = w_left[jx]
        for jx in range(len(s_right)):
            if s_right[jx] == SIL_WORD: s_right[jx] = ''
            if s_right[jx] == SELF_WORD: s_right[jx] = w_right[jx]
        for jx in range(len(c_s_words)):
            if c_s_words[jx] == SIL_WORD: c_s_words[jx] = ''
            if c_s_words[jx] == SELF_WORD: c_s_words[jx] = c_w_words[jx]

        # Extract input_words and output_words
        c_w_words = word_tokenize(' '.join(c_w_words))
        c_s_words = word_tokenize(' '.join(c_s_words))
        w_input = w_left + [EXTRA_ID_0] + c_w_words + [EXTRA_ID_1] + w_right
        input_words = [TN_PREFIX] + w_input
        output_words = c_s_words
        # Finalize
        self.input_str = ' '.join(input_words)
        self.output_str = ' '.join(output_words)
        self.semiotic_class = semiotic_class

class TextNormalizationDecoderDataset:
    def __init__(
            self,
            input_file: str,
            tokenizer: PreTrainedTokenizerBase,
            max_len: int,
            decoder_data_augmentation: bool
        ):
        raw_insts = read_data_file(input_file)

        # Convert raw instances to TaggerDataInstance
        insts, inputs, targets = [], [], []
        for (classes, w_words, s_words) in tqdm(raw_insts):
            for ix, (_class, w_word, s_word) in enumerate(zip(classes, w_words, s_words)):
                if not s_word in SPECIAL_WORDS:
                    inst = DecoderDataInstance(w_words, s_words, ix, ix+1, _class)
                    insts.append(inst)
                    if decoder_data_augmentation:
                        noise_left = random.randint(1, 2)
                        noise_right = random.randint(1, 2)
                        inst = DecoderDataInstance(w_words, s_words,
                                                   start_idx=ix-noise_left,
                                                   end_idx=ix+1+noise_right)
                        insts.append(inst)

        self.insts = insts
        inputs = [inst.input_str for inst in insts]
        targets = [inst.output_str for inst in insts]

        # Tokenization
        self.inputs, self.examples = [], []
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
            with tokenizer.as_target_tokenizer():
                _target = tokenizer([targets[idx]])
            target_len = len(_target['input_ids'][0])
            if target_len > max_len:
                long_examples_filtered += 1
                continue

            # Update
            self.inputs.append(inputs[idx])
            _input['labels'] = _target['input_ids']
            self.examples.append(_input)
            if inputs[idx].startswith(TN_PREFIX): self.tn_count += 1
            if inputs[idx].startswith(ITN_PREFIX): self.itn_count += 1
            input_max_len = max(input_max_len, input_len)
            target_max_len = max(target_max_len, target_len)
        print(f'long_examples_filtered: {long_examples_filtered}')
        print(f'input_max_len: {input_max_len} | target_max_len: {target_max_len}')

    def __getitem__(self, idx):
        example = self.examples[idx]
        item = {key: val[0] for key, val in example.items()}
        return item

    def __len__(self):
        return len(self.examples)
