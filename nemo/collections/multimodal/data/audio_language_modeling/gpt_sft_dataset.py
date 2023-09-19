# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset
from nemo.utils import logging

__all__ = ['MMGPTSFTDataset']

class MMGPTSFTDataset(GPTSFTDataset):
    """
    MultiModal GPT STF Dataset
    """
    def _audio_codes_to_text(self, audio_codes):
        """
        convert audio_codes to text readable by text tokenizer
        [0, 1, 2] -> "<audio_id_0><audio_id_1><audio_id_2>"
        turns out that tokenizing this type of string is very slow and not practical !!!
        """
        return "".join([f"<audio_id_{x}>" for x in audio_codes])

    def _load_audio_data(self, filepath):
        """
        load audio data from npz file
        """
        # make sure that it is npz file
        assert filepath.endswith('.npz'), f"audio_codes should be npz file, got {filepath}"

        AUDIO_ID_OFFSET = 256003        # @kpuvvada: move to config file
        audio_codes = np.load(filepath)['codes']
        audio_codes = np.squeeze(audio_codes)
        assert len(audio_codes.shape) == 1, f"audio_codes should be 1D, got {audio_codes.shape}"
        
        audio_codes = audio_codes + AUDIO_ID_OFFSET
        # return self._audio_codes_to_text(audio_codes)
        return audio_codes.tolist()


    # @kpuvvada: deprecated; not used - to be removed
    def _process_example_deprecated(self, example):
        """
        Overriding the _process_example method from GPTSFTDataset to handle audio inputs
        """

        # question = example['question']  # @kpuvvada: remove hardcoded key
        question = "[ Transcribe in English ] "
        context = example['audio_codes']
        output = example[self.label_key]

        if context.endswith('.npz'):
            context = self._load_audio_data(context)

        # @kpuvvada: making prompt template compulsary; remove later
        assert self.prompt_template is not None, "Prompt template is compulsary for multimodal GPT SFT"

        if self.prompt_template is not None:
            question = self.prompt_template[0] + question
            question = self.tokenizer.text_to_ids(question)
            context = question  + context
            context = context + self.tokenizer.text_to_ids(self.prompt_template[1])
            output = self.tokenizer.text_to_ids(output)

        if self.separate_prompt_and_response_with_newline and self.prompt_template is None:
            text = context + '\n' + output
        elif not self.separate_prompt_and_response_with_newline and self.prompt_template is None:
            text = context + ' ' + output
        
        # tokenized_text = self.tokenizer.text_to_ids(text)
        # context_ids = self.tokenizer.text_to_ids(context)
        # answer_ids = tokenized_text[len(context_ids) :]
        context_ids = context
        answer_ids = output

        total_ids = len(context_ids) + len(answer_ids)
        if self.add_bos:
            total_ids += 1
        if self.add_sep:
            total_ids += 1
        if self.add_eos:
            total_ids += 1

        # If the total number of token is greater than the max, we will try to truncate the answer
        if total_ids > self.max_seq_length:
            truncation_length = total_ids - self.max_seq_length
            if self.truncation_field == "answer":
                answer_ids = answer_ids[: -min(truncation_length, len(answer_ids))]
            elif self.truncation_field == "context":
                context_ids = context_ids[: -min(truncation_length, len(context_ids))]

        if len(context_ids) > self.max_seq_length:
            context_ids = context_ids[: self.max_seq_length]

        assert len(context_ids) <= self.max_seq_length
        input_ids = context_ids

        answer_start_idx = len(input_ids)
        # Adds sep token between text/prompt and answer
        if self.add_sep:
            input_ids = input_ids + [self.sep_id]
            answer_start_idx += 1

        input_ids = input_ids + answer_ids

        if self.add_bos:
            input_ids = [self.tokenizer.bos_id] + input_ids
            answer_start_idx += 1
        if self.add_eos:
            input_ids = input_ids + [self.tokenizer.eos_id]

        if len(input_ids) < self.min_seq_length or len(input_ids) > self.max_seq_length:
            input_ids = input_ids[: self.max_seq_length]

        # store metadata in dataset, in case user may have keys required in the prediction json files
        # metadata = {k: v for k, v in example.items() if k not in self.prompt_template_keys}
        metadata = {}       # @kpuvvada: temporary placeholder
        
        processed_example = {
            'input_ids': input_ids,
            'answer_start_idx': answer_start_idx,
            'context_ids': context_ids,
            'context_length': len(context_ids),
            'answer_ids': answer_ids,
            'metadata': metadata,
        }

        return processed_example


    def _process_example(self, example):
        """
        Overriding the _process_example method from GPTSFTDataset to handle audio inputs
        
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.
        """
        prompt_template_values = [example[c].strip(' ') for c in self.prompt_template_keys]

        template_strings, template_strings_keys = self._separate_template(prompt_template_values)
        template_ids = [self.tokenizer.text_to_ids(s) for s in template_strings]
        
        # Insert audio codes
        if 'audio_codes' in template_strings_keys:
            # find the respective index
            audio_codes_idx = template_strings_keys.index('audio_codes')
            audio_ids = self._load_audio_data(example['audio_codes'])
            template_ids[audio_codes_idx] = audio_ids
        
        context_ids, answer_ids = self._multiple_truncation(template_ids, template_strings_keys)

        if self.virtual_tokens:
            # (@adithyare) we are going to insert "pad/eos" tokens in the beginning of the text and context
            # these pad/eos tokens are placeholders for virtual tokens
            context_ids = [self.tokenizer.eos_id] * self.virtual_tokens + context_ids

        input_ids = context_ids
        answer_start_idx = len(input_ids)

        # Adds bos token in the start
        if self.add_bos:
            context_ids = [self.tokenizer.bos_id] + context_ids
            input_ids = [self.tokenizer.bos_id] + input_ids
            answer_start_idx += 1

        # Adds sep token between text/prompt and answer
        if self.add_sep:
            context_ids = context_ids + [self.sep_id]
            input_ids = input_ids + [self.sep_id]
            answer_start_idx += 1

        input_ids = input_ids + answer_ids

        # Only training need to consider eos token
        if self.add_eos:
            input_ids = input_ids + [self.tokenizer.eos_id]

        if len(input_ids) > self.max_seq_length:
            logging.warning(f'Input ids length {len(input_ids)} exceed max sequence length {self.max_seq_length}')
            input_ids = input_ids[: self.max_seq_length]

        # store metadata in dataset, in case user may have keys required in the prediction json files
        metadata = {k: v for k, v in example.items() if k not in self.prompt_template_keys}
        processed_example = {
            'input_ids': input_ids,
            'answer_start_idx': answer_start_idx,
            'context_ids': context_ids,
            'context_length': len(context_ids),
            'answer_ids': answer_ids,
            'metadata': metadata,
        }

        return processed_example