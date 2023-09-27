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
    ):
        """
        see GPTSFTDataset for description of non-audio args
        """
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


    # @kpuvvada: this is deprecated; not used - remove
    def _audio_codes_to_text(self, audio_codes):
        """
        convert audio_codes to text readable by text tokenizer
        [0, 1, 2] -> "<audio_id_0><audio_id_1><audio_id_2>"
        turns out that tokenizing this type of string is very slow and not practical !!!
        """
        return "".join([f"<audio_id_{x}>" for x in audio_codes])

    def _load_audio_data(self, filepath, num_codebooks_to_load=8, keep_dims_for_audio=True):
        """
        load audio data from npz file
        keep_dims_for_audio: if False, applies squeeze to audio_codes
        """
        # make sure that it is npz file
        assert filepath.endswith('.npz'), f"audio_codes should be npz file, got {filepath}"

        audio_codes = np.load(filepath)['codes'][:num_codebooks_to_load, :] # (num_codebooks, num_frames)
        audio_codes = np.transpose(audio_codes) # (num_frames, num_codebooks
        if not keep_dims_for_audio:
            audio_codes = np.squeeze(audio_codes)
        
        # we can now accomodate 2D audio codes, so the following is not needed
        # assert len(audio_codes.shape) == 1, f"audio_codes should be 1D, got {audio_codes.shape}"
        
        audio_codes = audio_codes + self.audio_token_offset
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
            audio_ids = self._load_audio_data(example['audio_codes'], num_codebooks_to_load=self.num_audio_codebooks, keep_dims_for_audio=True)
            template_ids[audio_codes_idx] = audio_ids
        
        # pad all entries to same dimension
        template_ids = self._pad_all_to_same_dims(template_ids, pad_id=self.tokenizer.pad_id)
        
        context_ids, answer_ids = self._multiple_truncation(template_ids, template_strings_keys)

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
        attention_mask = [self._create_attention_mask(max_length) for _ in batch]
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