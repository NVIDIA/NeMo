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


import re
from typing import List, Optional\

import numpy as np
import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_dataset import GPTSFTDataset
from nemo.utils import logging

__all__ = ['MMGPTSFTDataset']

class MMGPTSFTDataset(GPTSFTDataset):
    """
    MultiModal GPT STF Dataset
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerSpec,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        add_sep: bool = False,
        sep_id: int = None,
        max_num_samples: int = None,
        seed: int = 1234,
        label_key: str = "answer",
        answer_only_loss: bool = True,
        truncation_field: str = "text",
        pad_to_max_length: bool = False,  # (@adithyare) allows for much faster training especially in PEFT settings.
        index_mapping_dir: str = None,
        prompt_template: str = None,
        virtual_tokens: int = 0,
        tokens_to_generate: int = 0,
        memmap_workers: Optional[int] = None,
        hf_dataset: bool = False,
        truncation_method: str = 'right',
        num_audio_codebooks: int = 1,
        audio_codebook_size: int = 1024,
        audio_token_offset: int = 256003,
        pad_audio_to_length: int = 0,
        attn_mask_type: str = 'causal',
        task_templates: List[dict] = None,
    ):
        """
        see GPTSFTDataset for description of non-audio args
        """
        # one of prompt_template or task_templates should be provided
        # but not both
        assert (prompt_template is None) != (task_templates is None), "either prompt_template or task_templates should be provided"
        
        super().__init__(
            file_path=file_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            add_bos=add_bos,
            add_eos=add_eos,
            add_sep=add_sep,
            sep_id=sep_id,
            max_num_samples=max_num_samples,
            seed=seed,
            label_key=label_key,
            answer_only_loss=answer_only_loss,
            truncation_field=truncation_field,
            pad_to_max_length=pad_to_max_length,
            index_mapping_dir=index_mapping_dir,
            prompt_template=prompt_template,
            virtual_tokens=virtual_tokens,
            tokens_to_generate=tokens_to_generate,
            memmap_workers=memmap_workers,
            hf_dataset=hf_dataset,
            truncation_method=truncation_method,
        )

        self.num_audio_codebooks = num_audio_codebooks
        self.audio_codebook_size = audio_codebook_size
        self.audio_token_offset = audio_token_offset
        self.pad_audio_to_length = pad_audio_to_length
        self.attn_mask_type = attn_mask_type

        if task_templates is not None:
            self._load_task_templates(task_templates)
            self.prompt_template = None
            self.prompt_template_keys = None

    
    def _maybe_validate_prompt_template(self):
        if self.prompt_template is not None:
            # call parent class method
            super()._maybe_validate_prompt_template()
            
    
    def _load_task_templates(self, task_templates):
        """
        Takes in the task template portion of the config and turns  
        it into a table
        """
        
        def _get_prompt_template_keys(prompt_template):
            # When providing things like newlines in the prompt template via the CLI, they are escaped. This line unescapes them.
            prompt_template = prompt_template.encode('utf-8').decode('unicode_escape')
            prompt_template_keys = re.findall(r'{(.*?)}', prompt_template)
            return prompt_template_keys
        
        def _validate_task_template(task):
            # validate a single task
            assert (
                task.prompt_template is not None
            ), f"prompt_template is required for task {task.taskname}"

            answer_placeholder = f'{{{task.answer_key}}}'
            assert (
                task.prompt_template[-len(answer_placeholder) :] == answer_placeholder
            ), f'{answer_placeholder} must be at the end of prompt_template for task {task.taskname}'

            truncation_keys = task.truncation_keys.split(",") if task.truncation_keys is not None else []
            assert set(truncation_keys).issubset(
                _get_prompt_template_keys(task.prompt_template)
            ), f"truncation_keys should be subset of prompt_template_keys for task {task.taskname}"
        
        
        self.task_templates = {}
        self.task_id_num_to_name = {}
        task_id_num = 0
        for task in task_templates:
            # validate
            _validate_task_template(task)

            self.task_templates[task.taskname] = {
                "prompt_template": task.prompt_template,
                "prompt_template_keys": _get_prompt_template_keys(task.prompt_template),
                "answer_key": task.get("answer_key", None),
                "truncation_keys": task.truncation_keys.split(",") if task.truncation_keys is not None else [],
                "audio_keys" : task.audio_keys.split(",") if task.audio_keys is not None else [],
                "task_id_num": task_id_num,
            }
            self.task_id_num_to_name[task_id_num] = task.taskname
            task_id_num += 1
    
    # @kpuvvada: this is deprecated; not used - remove
    def _audio_codes_to_text(self, audio_codes):
        """
        convert audio_codes to text readable by text tokenizer
        [0, 1, 2] -> "<audio_id_0><audio_id_1><audio_id_2>"
        turns out that tokenizing this type of string is very slow and not practical !!!
        """
        return "".join([f"<audio_id_{x}>" for x in audio_codes])

    def _load_audio_data(
            self, 
            filepath, 
            num_codebooks_to_load=8, 
            keep_dims_for_audio=True, 
            pad_to_length=1500, 
            pad_location='end',
            pad_id=None,
            truncate_to_length=0,
    ):
        """
        load audio data from npz file
        keep_dims_for_audio: if False, applies squeeze to audio_codes
        pad_to_length:  if = 0 do nothing; 
                        if > 0 pads audio_codes to this length
                        
        """
        # make sure that both pad_to_length and truncate_to_length are not set
        assert not (pad_to_length > 0 and truncate_to_length > 0), "pad_to_length and truncate_to_length cannot be set at the same time"

        # make sure that it is npz file
        assert filepath.endswith('.npz'), f"audio_codes should be npz file, got {filepath}"
        assert pad_location in ['begin', 'end'], f"pad_location should be either 'start' or 'end', got {pad_location}"

        audio_codes = np.load(filepath)['codes'][:num_codebooks_to_load, :] # (num_codebooks, num_frames)
        audio_codes = np.transpose(audio_codes) # (num_frames, num_codebooks)
        if not keep_dims_for_audio:
            audio_codes = np.squeeze(audio_codes)
        
        # we can now accomodate 2D audio codes, so the following is not needed
        # assert len(audio_codes.shape) == 1, f"audio_codes should be 1D, got {audio_codes.shape}"
        
        audio_codes = audio_codes + self.audio_token_offset
        # return self._audio_codes_to_text(audio_codes)

        # pad to length after adding offset so that audio masking works as expected
        if pad_to_length > 0:
            assert pad_id is not None, "pad_id should be provided if pad_to_length is > 0"
            pad_len = pad_to_length - audio_codes.shape[0]
            if pad_len > 0:
                pad_matrix = np.ones((pad_len, audio_codes.shape[1]), dtype=audio_codes.dtype) * pad_id
                if pad_location == 'end':
                    audio_codes = np.concatenate((audio_codes, pad_matrix), axis=0)
                else:
                    audio_codes = np.concatenate((pad_matrix, audio_codes), axis=0)

        if truncate_to_length > 0:
            if truncate_to_length < audio_codes.shape[0]:
                audio_codes = audio_codes[:truncate_to_length, :]

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


    def _pad_all_to_same_dims(self, template_ids, pad_id):
        """
        pad all entries of template_ids to same dimension (2D for now)
        """
        # the following is non-optimal 
        # too many conversions between list and np.array
        # @kpuvvada: fix this later

        for i, item in enumerate(template_ids):
            item = np.array(item)
            if len(item.shape) == 2:        # assmusing that all are 2D tensors if not 1D
                continue
            else:
                item = np.expand_dims(item, axis=-1)
                seq_len = item.shape[0]
                pad_matrix = np.ones((seq_len, self.num_audio_codebooks -1 ), dtype=item.dtype) * pad_id      # num_codebooks -1 is hardcoded. should be fine for audio and text
                item = np.concatenate((item, pad_matrix), axis=-1)
                template_ids[i] = item.tolist()
        return template_ids
    

    def _filter_template_strings_and_keys(self, template_strings, template_strings_keys, not_in_example_value):
        remove_indices = []
        # find indices of template_strings that are not in example
        for i, item in enumerate(template_strings):
            if item == not_in_example_value:
                remove_indices.extend([i-1, i])

        # remove those indices from both template_strings and template_strings_keys
        template_strings = [item for i, item in enumerate(template_strings) if i not in remove_indices]
        template_strings_keys = [item for i, item in enumerate(template_strings_keys) if i not in remove_indices]
        return template_strings, template_strings_keys

    def _swap_template_ids_for_audio(self, template_ids, template_strings_keys, example, field_name=None):
        """
        replace template_ids with audio_ids
        """
        pad_location = 'end' # pad at beginning or end, hardcoded for now
        pad_to_length = self.pad_audio_to_length
        pad_id = self.tokenizer.eos_id if pad_location=='end' else self.tokenizer.pad_id
        truncate_to_length = 7*75 if 'prompt' in field_name else 0   # hardcoded for now

        # Insert audio codes
        if field_name in template_strings_keys:
            # find the respective index
            audio_codes_idx = template_strings_keys.index(field_name)
            audio_ids = self._load_audio_data(
                example[field_name], 
                num_codebooks_to_load=self.num_audio_codebooks, 
                keep_dims_for_audio=True, 
                pad_to_length=pad_to_length,
                pad_location=pad_location, 
                pad_id=pad_id,
                truncate_to_length=truncate_to_length,
            )
            template_ids[audio_codes_idx] = audio_ids

        return template_ids


    def _separate_template(self, prompt_template_values: List[str], prompt_template_keys: List[str], prompt_template: str):
        """
        Combine contexts and label based on prompt_template into a list of strings and a list of keys.

        Args:
            prompt_template_values (List[str]): the list of context and label strings extrated from jsonl file with prompt_template_keys.

        Returns:
            template_strings (List[str]): separated prompt_template with contexts/label placeholder filled with corresponding strings
            template_strings_keys (List[str]): strings point to placeholder keys or <template>
            
        Examples:
            see __super__._separate_template
        """
        placeholders = [f'{{{k}}}' for k in prompt_template_keys]

        # placeholder to string
        ph_to_s = {ph: s for ph, s in zip(placeholders, prompt_template_values)}
        # placeholder to key
        ph_to_k = {ph: k for ph, k in zip(placeholders, prompt_template_keys)}

        # separate prompt_template based on '<space>{placeholder}'
        # examples:
        #   self.prompt_template = "Context:{context}  Passage: {passage}\n\nQuestion:{question} {label}"
        #   template_with_placeholder_separated = ['Context:', '{context}', '  Passage:', ' {passage}', '\n\nQuestion:', '{question}', ' {label}']
        template_with_placeholder_separated = re.split('( *?{.+?})', prompt_template)
        template_with_placeholder_separated = [s for s in template_with_placeholder_separated if len(s) > 0]

        # remove space if we have leading space and tokenizer is not space_sensitive
        # space_sensitive = True : tokenizer.text_to_tokens('A{num_spaces}B') = tokenizer.text_to_tokens('A') + tokenizer.text_to_tokens('{num_spaces}B')
        # space_sensitive = False: tokenizer.text_to_tokens('A{num_spaces}B') = tokenizer.text_to_tokens('A') + tokenizer.text_to_tokens('{num_spaces-1}B')
        space_sensitive = getattr(self.tokenizer, 'space_sensitive', False)
        template_with_space_reduced = [
            s[1:] if not space_sensitive and s[0] == ' ' else s for s in template_with_placeholder_separated
        ]

        # convert placeholder to the corresponding string (preserve left spaces) and key
        template_strings, template_strings_keys = [], []
        for t in template_with_space_reduced:
            placeholder = t.lstrip(' ')
            left_spaces = ' ' * (len(t) - len(placeholder))
            template_strings.append(left_spaces + ph_to_s.get(placeholder, placeholder))
            template_strings_keys.append(ph_to_k.get(placeholder, '<template>'))

        return template_strings, template_strings_keys
    

    def _multiple_truncation(self, template_ids: List[List[int]], template_ids_keys: List[str], truncation_fields: List[str]):
        """
        Calculate total tokens and truncate multiple contexts in truncation_fields.
        
        Args:
            template_ids (List[List[int]]): the list of separate prompt_template ids.
            template_ids_keys (List[str]): the list of placeholder keys or <template> (used to check key in truncation_fields).

        Returns:
            context_ids (List[int]): all context ids.
            label_ids (List[int]): all label ids.
        """
        context_ids = template_ids[:-1]
        label_ids = template_ids[-1]
        total_ids = (
            self.virtual_tokens
            + sum(len(ids) for ids in context_ids)
            + max(len(label_ids), self.tokens_to_generate)
            + self.add_bos
            + self.add_sep
            + self.add_eos  # Only training need to consider eos token
        )

        if total_ids > self.max_seq_length:
            truncation_length_total = total_ids - self.max_seq_length
            num_fields = len(truncation_fields)
            # sorted equal divide length to each field
            # examples:
            #   truncation_length_total = 3
            #   num_fields = 11
            #   truncation_length_list = [3,4,4]
            truncation_length_list = [
                truncation_length_total // num_fields + (1 if i < truncation_length_total % num_fields else 0)
                for i in range(num_fields)[::-1]
            ]

            for i, (ids, key) in enumerate(zip(template_ids, template_ids_keys)):
                if key in truncation_fields:
                    truncation_length = truncation_length_list.pop()
                    assert len(ids) >= truncation_length, f'{key} is not long enough to truncate.'
                    if self.truncation_method == 'left':
                        window_offset = truncation_length
                    elif self.truncation_method == 'right':
                        window_offset = 0
                    else:
                        raise ValueError(f'{self.truncation_method} is not supported')

                    window_length = len(ids) - truncation_length
                    template_ids[i] = ids[window_offset : window_offset + window_length]

        context_ids = [i for ids in template_ids[:-1] for i in ids]
        label_ids = template_ids[-1]
        return context_ids, label_ids
    
    
    def _process_example(self, example):
        """
        Overriding the _process_example method from GPTSFTDataset to handle audio inputs
        
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.
        """
        # currently keeping both prompt_template and task_templates
        # deprecate prompt_template later
        if self.task_templates is not None:
            taskname = example.get('taskname', None)
            assert (
                taskname is not None
            ), f"taskname is missing for example {example}"

            prompt_template = self.task_templates[taskname]['prompt_template']
            prompt_template_keys = self.task_templates[taskname]['prompt_template_keys']
            audio_keys = self.task_templates[taskname]['audio_keys']
            truncation_keys = self.task_templates[taskname]['truncation_keys']
        
        else: # prompt_template is not None
            prompt_template = self.prompt_template
            prompt_template_keys = self.prompt_template_keys
            audio_keys = ['audio_codes', 'speaker_prompt', 'audio_codes1', 'audio_codes2']   # hardcoded for now
            truncation_keys = self.truncation_fields

        prompt_template_values = [example[c].strip(' ') for c in prompt_template_keys]
        
        # following is a hack to make it work for prompt_template_keys not in examples
        # not_in_example_value = 'notInExample1234'
        # prompt_template_values = [example.get(c, not_in_example_value).strip(' ') for c in self.prompt_template_keys]
        # hack is not needed any more since we are using task_templates

        template_strings, template_strings_keys = self._separate_template(prompt_template_values, prompt_template_keys, prompt_template)
        
        # not needed as we are using task_templates
        # template_strings, template_strings_keys = self._filter_template_strings_and_keys(template_strings, template_strings_keys, not_in_example_value)
        
        template_ids = [self.tokenizer.text_to_ids(s) for s in template_strings]
        
        # Insert audio codes
        for key in audio_keys:
            template_ids = self._swap_template_ids_for_audio(template_ids, template_strings_keys, example, field_name=key)
        # template_ids = self._swap_template_ids_for_audio(template_ids, template_strings_keys, example, field_name='audio_codes')
        # template_ids = self._swap_template_ids_for_audio(template_ids, template_strings_keys, example, field_name='speaker_prompt')
        # template_ids = self._swap_template_ids_for_audio(template_ids, template_strings_keys, example, field_name='audio_codes1')
        # template_ids = self._swap_template_ids_for_audio(template_ids, template_strings_keys, example, field_name='audio_codes2')
        
        """ moved to _swap_template_ids_for_audio
        pad_location = 'end' # pad at beginning or end, hardcoded for now
        if 'audio_codes' in template_strings_keys:
            # find the respective index
            audio_codes_idx = template_strings_keys.index('audio_codes')
            audio_ids = self._load_audio_data(
                example['audio_codes'], 
                num_codebooks_to_load=self.num_audio_codebooks, 
                keep_dims_for_audio=True, 
                pad_to_length=self.pad_audio_to_length,
                pad_location=pad_location, 
                pad_id=self.tokenizer.eos_id if pad_location=='end' else self.tokenizer.pad_id,
            )
            template_ids[audio_codes_idx] = audio_ids
        """

        # pad all entries to same dimension
        template_ids = self._pad_all_to_same_dims(template_ids, pad_id=self.tokenizer.pad_id)
        
        context_ids, answer_ids = self._multiple_truncation(template_ids, template_strings_keys, truncation_fields=truncation_keys)

        if self.virtual_tokens:
            # (@adithyare) we are going to insert "pad/eos" tokens in the beginning of the text and context
            # these pad/eos tokens are placeholders for virtual tokens
            context_ids = [[self.tokenizer.eos_id] * self.num_audio_codebooks] * self.virtual_tokens + context_ids

        input_ids = context_ids
        answer_start_idx = len(input_ids)

        # Adds bos token in the start
        if self.add_bos:
            context_ids = [[self.tokenizer.bos_id] * self.num_audio_codebooks ]+ context_ids
            input_ids = [[self.tokenizer.bos_id] * self.num_audio_codebooks ]+ input_ids
            answer_start_idx += 1

        # Adds sep token between text/prompt and answer
        if self.add_sep:
            context_ids = context_ids + [[self.sep_id] * self.num_audio_codebooks]
            input_ids = input_ids + [[self.sep_id] * self.num_audio_codebooks]
            answer_start_idx += 1

        input_ids = input_ids + answer_ids

        # Only training need to consider eos token
        if self.add_eos:
            input_ids = input_ids + [[self.tokenizer.eos_id] * self.num_audio_codebooks]

        if len(input_ids) > self.max_seq_length:
            logging.warning(f'Input ids length {len(input_ids)} exceed max sequence length {self.max_seq_length}')
            input_ids = input_ids[: self.max_seq_length]

        # store metadata in dataset, in case user may have keys required in the prediction json files
        metadata = {k: v for k, v in example.items() if k not in prompt_template_keys}
        processed_example = {
            'input_ids': input_ids,
            'answer_start_idx': answer_start_idx,
            'context_ids': context_ids,
            'context_length': len(context_ids),
            'answer_ids': answer_ids,
            'metadata': metadata,
        }

        return processed_example

    
    @torch.no_grad()
    def _create_attention_mask(self, max_length, context_length=None):
        """Create `attention_mask`.
        Args:
            input_ids: A 1D tensor that holds the indices of tokens.
        """
        # seq_length = len(input_ids)
        # `attention_mask` has the shape of [1, seq_length, seq_length]
        attention_mask = torch.tril(torch.ones((max_length, max_length))).unsqueeze(0)
        attention_mask = attention_mask < 0.5

        if self.attn_mask_type == 'causal':
            return attention_mask
        elif self.attn_mask_type == 'prefix':
            assert context_length is not None, "context_length is required for prefix attention mask"

            attention_mask[:, :context_length, :context_length] = False
            return attention_mask
        else:
            raise NotImplementedError(f"attn_mask_type {self.attn_mask_type} not implemented")
    
    
    def _collate_item(self, item, max_length, pad_id):
        item = self._maybe_cast_to_list(item)
        # max_length = max([len(x) for x in item]) if item else 0
        # here [0] should be tokenizer.pad_id
        if isinstance(item[0][0], list):
            # list per timestep
            item = [x + [[pad_id]* self.num_audio_codebooks] * (max_length - len(x)) for x in item]
        else:
            # int per timestep
            item = [x + [pad_id] * (max_length - len(x)) for x in item]
        return item
    
    
    def collate_fn(self, batch):
        input_ids = [item['input_ids'][:-1] for item in batch]
        labels = [item['input_ids'][1:] for item in batch]
        contexts = [item['context_ids'] for item in batch]
        context_lengths = torch.LongTensor([item['context_length'] for item in batch])
        answers = [item['answer_ids'] for item in batch]
        loss_mask = [self._build_loss_mask(item)[1:] for item in batch]
        metadata = [item['metadata'] for item in batch]

        max_length = max(max([len(x) for x in input_ids]), max([len(x) for x in contexts]) + self.tokens_to_generate)
        # increase max length to nearest multiple of 4 or 8
        if self.pad_to_max_length:
            max_length = self.max_seq_length
        else:
            max_length = min(self.max_seq_length, self._ceil_to_nearest(max_length, 8))
        assert max_length <= self.max_seq_length
        # attention_mask = [self._create_attention_mask(max_length) for _ in batch]
        attention_mask = [self._create_attention_mask(max_length, context_lengths[i].item()) for i, _ in enumerate(batch)]
        attention_mask = torch.stack(attention_mask)
        position_ids = [list(range(max_length)) for _ in batch]
        position_ids = torch.LongTensor(position_ids)
        input_ids = torch.LongTensor(
            self._collate_item(input_ids, max_length=max_length, pad_id=self.tokenizer.eos_id)
        )
        labels = torch.LongTensor(self._collate_item(labels, max_length=max_length, pad_id=self.tokenizer.eos_id))
        loss_mask = torch.LongTensor(self._collate_item(loss_mask, max_length=max_length, pad_id=0))
        contexts = torch.LongTensor(self._collate_item(contexts, max_length=max_length, pad_id=self.tokenizer.eos_id))
        answers = torch.LongTensor(self._collate_item(answers, max_length=max_length, pad_id=self.tokenizer.eos_id))

        # pass first codebook through regular GPT pipeline and rest through audio pipeline
        # GPT related batch
        processed_batch = {
            'tokens': input_ids[:,:, 0],
            'labels': labels[:,:, 0],
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'contexts': contexts[:,:, 0],
            'context_lengths': context_lengths,
            'answers': answers[:,:, 0],
            'metadata': metadata,
        }

        # audio related batch
        if self.num_audio_codebooks >1 :
            processed_batch_audio = {
                'additional_tokens': input_ids[:,:, 1:],    
                'additional_tokens_mask': (input_ids[:,:, 1:] >= self.audio_token_offset).float(), # TODO: is this the correct dtype
                'additional_contexts': contexts[:,:, 1:],       # have to include additional_labels when doing TTS, ignoring for now
                'additional_contexts_mask': (contexts[:,:, 1:] >= self.audio_token_offset).float(),
            }
        else:
            processed_batch_audio = {
                'additional_tokens': torch.empty(0, dtype=input_ids.dtype),
                'additional_tokens_mask': torch.empty(0, dtype=torch.float32),      # may be change this dtype to model/trainer dtype? 
                'additional_contexts': torch.empty(0, dtype=contexts.dtype),
                'additional_contexts_mask': torch.empty(0, dtype=torch.float32),
            }

        # merge both 
        processed_batch.update(processed_batch_audio)

        return processed_batch