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
        AUDIO_ID_OFFSET = 256003
        audio_codes = np.load(filepath)['codes']
        audio_codes = np.squeeze(audio_codes)
        assert len(audio_codes.shape) == 1, f"audio_codes should be 1D, got {audio_codes.shape}"
        
        audio_codes = audio_codes + AUDIO_ID_OFFSET
        # return self._audio_codes_to_text(audio_codes)
        return audio_codes.tolist()


    def _process_example(self, example):
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
