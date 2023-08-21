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

import json
import os
import pickle

import torch
from tqdm.auto import tqdm
import numpy as np

from nemo.collections.nlp.modules.common import VirtualPromptSource
from nemo.collections.nlp.data.language_modeling.megatron.base_prompt_learning_dataset import BasePromptLearningDataset
from nemo.collections.nlp.data.language_modeling.megatron.retro_fine_tune_dataset import RetroQAFineTuneDataset
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import JSONLMemMapDataset

from nemo.collections.nlp.modules.common.megatron.utils import build_position_ids
from nemo.core import Dataset
from nemo.utils import AppState, logging

__all__ = ['RetroPromptLearningDataset']


class RetroPromptLearningDataset(RetroQAFineTuneDataset, BasePromptLearningDataset):
    """
    The dataset class for prompt-tuning or p-tuning pretrained GPT models.
    
    Args:
        data (list[strings], list[dicts]): (1) paths to .jsonl or .json files, (2) dict objects corresponding to each input example
        tokenizer (tokenizer): Tokenizer from frozen language model
        virtual_prompt_source (Enum): Either VirtualPromptSource.PROMPT_TABLE or VirtualPromptSource.PROMPT_ENCODER
        task_templates (dict): Dictionary containing all task template information needed to format prompts. Created in the GPTPromptLearningModel class.
        pseudo_tokens (list[strings]): A list of virtual prompt token placeholders e.g [<prompt_1>, <prompt_2>, ...] up to max num virtual tokens
        pad_token_id (int): ID of pad token from tokenizer
        max_seq_length (int): maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset. Data examples will be dropped if they do not meet the min length requirements. 
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add an end of sentence token to each data example
        for_train (bool): Whether you're creating a dataset for training or inference
        tokens_to_generate (int): (inference only) Number of tokens to generate during inference
    """

    def __init__(
        self,
        data,
        tokenizer,
        virtual_prompt_source: VirtualPromptSource,
        task_templates: dict,
        pseudo_tokens,
        pad_token_id: int,
        max_seq_length: int,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        for_train: bool = True,
        tokens_to_generate=None,
        cache_data_path: str = None,  # the cache file
        load_cache: bool = True,  # whether to load from the cache if it is available
        seed: int = 1234,
        neighbors: int = 2,
        max_num_samples: int = None,
        megatron_lm_compatible: bool = False,
        retrieved_doc_len: int = 128,
        chunk_size: int = 64
    ):
        self.tokenizer = tokenizer
        self.virtual_prompt_source = virtual_prompt_source
        self.task_templates = task_templates
        self.pseudo_tokens = pseudo_tokens
        self.pseudo_token_ids = set(self.tokenizer.tokens_to_ids(self.pseudo_tokens))
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.for_train = for_train
        self.max_num_samples = max_num_samples
        self.neighbors = neighbors
        self.seed = seed
        self.megatron_lm_compatible = megatron_lm_compatible
        self.retrieved_doc_len = retrieved_doc_len
        self.chunk_size = chunk_size
        # self.examples = []

        if not self.for_train:
            self.tokens_to_generate = tokens_to_generate

        assert self.min_seq_length <= max_seq_length, "Min sequence length should be less than or equal to max"
        assert self.max_seq_length > 0, "Max sequence length should be greater than 0"

        logging.info("Loading and tokenizing dataset ... ")

        filename = self.check_data(data)
        self.indexed_dataset = JSONLMemMapDataset(dataset_paths=[filename], tokenizer=None, header_lines=0, workers=12)
        # Will be None after this call if `max_num_samples` is None
        self._build_samples_mapping(filename)


        # if load_cache and cache_data_path is not None and os.path.exists(cache_data_path):
        #     # load it from the cache
        #     logging.info(f'load the data from the cache file {cache_data_path}')
        #     with open(cache_data_path, 'rb') as f:
        #         self.examples = pickle.load(f)
        # else:
        #     # Data is just a list of dicts already loaded from a json file or passed in directly as a dict
        #     if isinstance(data[0], dict):
        #         self.load_data(data)

        #     # Datasets are a list of file path strings to .json or .jsonl files
        #     elif isinstance(data[0], str):
        #         for path in data:
        #             dataset = open(path, 'r', encoding='utf-8')
        #             self.load_data(dataset)
        #     else:
        #         raise ValueError("Datasets must be a list of filepath strings or a list of data example dicts")
        #     if cache_data_path is not None:
        #         # the first worker save the results into the cache file
        #         app_state = AppState()
        #         if app_state._global_rank == 0:
        #             with open(cache_data_path, 'wb') as f:
        #                 pickle.dump(self.examples, f)
        #             logging.info(f'save the data to the cache file {cache_data_path}')

    def check_data(self, filepath):
        """
        Loads a dataset by filling in the task templates specified in the config file
        with the information from each training/inference example. Converts all input 
        text into token ids. Also replaces the <|VIRTUAL_PROMPT_#|> placeholders in 
        the task templates with the actual virtual prompt token ids. 

        params:
            dataset: A list of json objects or a dictionary objects each
                     containing the information needed for a training example
        """
        skipped = 0
        dataset = open(filepath, 'r')
        tmp_file_name = filepath+'_checked.jsonl'
        tmp_file = open(tmp_file_name,'w')
        for json_line in tqdm(dataset):

            # Read example dict or load the information for a single example from .json file
            if type(json_line) == dict:
                doc = json_line
            else:
                doc = json.loads(json_line)

            taskname = doc["taskname"]
            prompt_template = self.task_templates[taskname]["prompt_template"]
            prompt_template_fields = self.task_templates[taskname]["prompt_template_fields"]
            total_virtual_tokens = self.task_templates[taskname]["total_virtual_tokens"]
            virtual_token_splits = self.task_templates[taskname]["virtual_token_splits"]
            truncation_field = self.task_templates[taskname]['truncate_field']
            answer_only_loss = self.task_templates[taskname]["answer_only_loss"]
            answer_field = self.task_templates[taskname]["answer_field"]

            input_example = prompt_template

            self._input_sanity_checks(
                total_virtual_tokens,
                virtual_token_splits,
                prompt_template,
                prompt_template_fields,
                truncation_field,
                answer_only_loss,
                answer_field,
                doc,
            )

            # Format the input example according to the template
            # input_example = " " + self._insert_text_in_template(input_example, prompt_template_fields, example) + " "
            input_example = input_example.replace("<|VIRTUAL_PROMPT_0|>", "").strip()
            input_example = self._insert_text_in_template(input_example, prompt_template_fields, doc)
            input_ids = self.tokenizer.text_to_ids(input_example)

            chunks = []
            contexts = doc['ctxs'] # are these neighbors ordered???????????????????????????????????????????????
            assert self.neighbors <= len(
                contexts
            ), f"specify {self.neighbors}, but only provide {len(contexts)} neighbors in the dataset"
            for i, neighbor in enumerate(contexts[: self.neighbors]):
                tokens = self.tokenizer.text_to_ids(neighbor)

                if i == 0:
                    input_ids = tokens + input_ids
                else:
                    tokens = tokens[:self.retrieved_doc_len]
                    if len(tokens) < self.retrieved_doc_len:
                        tokens = tokens + [self.pad_token_id] * (self.retrieved_doc_len - len(tokens))
                    chunks.append(tokens)



            # Add BOS/EOS if desired, adds EOS by default
            if self.add_bos:
                input_ids = [self.tokenizer.bos_id] + input_ids
            if self.add_eos:
                input_ids = input_ids + [self.tokenizer.eos_id]
            
            # these will be lobbed during the collate_fn
            temp_pads = list(self.pseudo_token_ids) 
            assert len(temp_pads) == total_virtual_tokens
            input_ids = temp_pads + input_ids

            answer_start_idx = len(input_ids)
            if answer_only_loss and self.for_train:
                answer_start_idx = self._find_answer_start(taskname, input_ids, answer_field, doc)
            
            length_before_answer = answer_start_idx
            
            # padding strategy 1: consider virtual prompt length and make the length_before_answer be >= chunk size
            # padding_length = 0
            # if length_before_answer < self.chunk_size:
            #     padding_length = self.chunk_size - length_before_answer
            #     input_ids = input_ids[:len(temp_pads)] + [self.pad_token_id] * padding_length + input_ids[len(temp_pads):]
            #     answer_start_idx += padding_length
            
            # padding strategy 2:  make the length_before_answer be a mutiple of chunk_size
            # but still padding in the middle between virtual tokens and top1
            padding_length = (self.chunk_size - length_before_answer % self.chunk_size) % self.chunk_size
            input_ids = input_ids[:len(temp_pads)] + [self.pad_token_id] * padding_length + input_ids[len(temp_pads):]
            answer_start_idx += padding_length

            # Try to truncate input text to fit into the max sequence length
            if len(input_ids) > self.max_seq_length:
                input_ids = self._truncate_input(
                    truncation_field,
                    input_ids,
                    taskname,
                    doc,
                    prompt_template,
                    prompt_template_fields,
                    virtual_token_splits,
                )

            # Skip example if the final length doesn't fit length requirements even after truncation
            if self.min_seq_length <= len(input_ids) <= self.max_seq_length:
                
                tmp_file.write(json.dumps(doc) + '\n')
            else:
                skipped += 1

        tmp_file.close()
        self.answer_only_loss = answer_only_loss
        logging.info(f'Skipped {skipped} sentences, sequence length too short or too long even after truncation')
        return tmp_file_name
    
    def _process_example(self, example):
        
        taskname = example["taskname"]
        prompt_template = self.task_templates[taskname]["prompt_template"]
        prompt_template_fields = self.task_templates[taskname]["prompt_template_fields"]
        total_virtual_tokens = self.task_templates[taskname]["total_virtual_tokens"]
        virtual_token_splits = self.task_templates[taskname]["virtual_token_splits"]
        truncation_field = self.task_templates[taskname]['truncate_field']
        answer_only_loss = self.task_templates[taskname]["answer_only_loss"]
        answer_field = self.task_templates[taskname]["answer_field"]

        input_example = prompt_template

        
        # Format the input example according to the template
        input_example = input_example.replace("<|VIRTUAL_PROMPT_0|>", "").strip()
        input_example = self._insert_text_in_template(input_example, prompt_template_fields, example)
        input_ids = self.tokenizer.text_to_ids(input_example)


        chunks = []
        contexts = example['ctxs'] # are these neighbors ordered???????????????????????????????????????????????
        assert self.neighbors <= len(
            contexts
        ), f"specify {self.neighbors}, but only provide {len(contexts)} neighbors in the dataset"
        for i, neighbor in enumerate(contexts[: self.neighbors]):
            tokens = self.tokenizer.text_to_ids(neighbor)
            
            if i == 0: # prepend top 1 
                input_ids = tokens + input_ids
            else:
                tokens = tokens[:self.retrieved_doc_len]
                if len(tokens) < self.retrieved_doc_len:
                    tokens = tokens + [self.pad_token_id] * (self.retrieved_doc_len - len(tokens))
                chunks.append(tokens)

        # Add BOS/EOS if desired, adds EOS by default
        if self.add_bos:
            input_ids = [self.tokenizer.bos_id] + input_ids
        if self.add_eos:
            input_ids = input_ids + [self.tokenizer.eos_id]


        # these will be lobbed during the collate_fn
        temp_pads = list(self.pseudo_token_ids) 
        assert len(temp_pads) == total_virtual_tokens
        input_ids = temp_pads + input_ids

        
        answer_start_idx = len(input_ids)
        if answer_only_loss and self.for_train:
            answer_start_idx = self._find_answer_start(taskname, input_ids, answer_field, example)
         
        length_before_answer = answer_start_idx
        
        # padding strategy 1: consider virtual prompt length and make the length_before_answer be >= chunk size
        # padding_length = 0
        # if length_before_answer < self.chunk_size:
        #     padding_length = self.chunk_size - length_before_answer
        #     input_ids = input_ids[:len(temp_pads)] + [self.pad_token_id] * padding_length + input_ids[len(temp_pads):]
        #     answer_start_idx += padding_length
        
        # padding strategy 2:  make the length_before_answer be a mutiple of chunk_size
        # but still padding in the middle between virtual tokens and top1
        # padding_length = (self.chunk_size - length_before_answer % self.chunk_size) % self.chunk_size
        # input_ids = input_ids[:len(temp_pads)] + [self.pad_token_id] * padding_length + input_ids[len(temp_pads):]
        # answer_start_idx += padding_length

        # padding strategy 3: pad at the beginning, make the length_before_answer be a mutiple of chunk_size
        padding_length = (self.chunk_size - length_before_answer % self.chunk_size) % self.chunk_size
        input_ids = [self.pad_token_id] * padding_length + input_ids
        answer_start_idx += padding_length


        # Try to truncate input text to fit into the max sequence length
        if len(input_ids) > self.max_seq_length:
            input_ids = self._truncate_input(
                truncation_field,
                input_ids,
                taskname,
                example,
                prompt_template,
                prompt_template_fields,
                virtual_token_splits,
            )

       

        

        # answer_start_idx = len(tokenized_input)
        # input_ids = tokenized_input + target
        # assert len(input_ids) <= 128, "cannot handle more than two chunks yet"

        chunks = np.array(chunks).reshape(1, self.neighbors-1, -1).astype(np.int64)  # self.neighbors-1 because we prepend top1 before context
        repeat = max(1, (len(input_ids)-1 - 64) // 64 +1) # -1 because encoder shift one
        repeated_chunks = np.repeat(chunks, repeat, axis=0)

        results = (input_ids, answer_start_idx, repeated_chunks, total_virtual_tokens)
        return results

    '''
    def load_data(self, dataset):
        """
        Loads a dataset by filling in the task templates specified in the config file
        with the information from each training/inference example. Converts all input 
        text into token ids. Also replaces the <|VIRTUAL_PROMPT_#|> placeholders in 
        the task templates with the actual virtual prompt token ids. 

        params:
            dataset: A list of json objects or a dictionary objects each
                     containing the information needed for a training example
        """
        skipped = 0

        for json_line in tqdm(dataset):

            # Read example dict or load the information for a single example from .json file
            if type(json_line) == dict:
                doc = json_line
            else:
                doc = json.loads(json_line)

            taskname = doc["taskname"]
            prompt_template = self.task_templates[taskname]["prompt_template"]
            prompt_template_fields = self.task_templates[taskname]["prompt_template_fields"]
            total_virtual_tokens = self.task_templates[taskname]["total_virtual_tokens"]
            virtual_token_splits = self.task_templates[taskname]["virtual_token_splits"]
            truncation_field = self.task_templates[taskname]['truncate_field']
            answer_only_loss = self.task_templates[taskname]["answer_only_loss"]
            answer_field = self.task_templates[taskname]["answer_field"]

            input_example = prompt_template

            self._input_sanity_checks(
                total_virtual_tokens,
                virtual_token_splits,
                prompt_template,
                prompt_template_fields,
                truncation_field,
                answer_only_loss,
                answer_field,
                doc,
            )

            # Format the input example according to the template
            input_example = self._insert_text_in_template(input_example, prompt_template_fields, doc)
            input_example = self._insert_virtual_token_placeholders(input_example, virtual_token_splits)
            input_ids = self.tokenizer.text_to_ids(input_example)

            # Add BOS/EOS if desired, adds EOS by default
            if self.add_bos:
                input_ids = [self.tokenizer.bos_id] + input_ids
            if self.add_eos:
                input_ids = input_ids + [self.tokenizer.eos_id]

            # Try to truncate input text to fit into the max sequence length
            if len(input_ids) > self.max_seq_length:
                input_ids = self._truncate_input(
                    truncation_field,
                    input_ids,
                    taskname,
                    doc,
                    prompt_template,
                    prompt_template_fields,
                    virtual_token_splits,
                )

            # Skip example if the final length doesn't fit length requirements even after truncation
            if self.min_seq_length <= len(input_ids) <= self.max_seq_length:
                if self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
                    taskname_id = self.tokenizer.text_to_ids(taskname)

                elif self.virtual_prompt_source == VirtualPromptSource.PROMPT_TABLE:
                    taskname_id = self.task_templates[taskname]["task_id_num"]

                elif self.virtual_prompt_source == VirtualPromptSource.NO_PROMPT:
                    taskname_id = -1
                else:
                    raise ValueError("Invalid virtual prompt source specified")

                # Find answer field indices if training and answer_only_loss is True
                answer_start_idx = None
                if answer_only_loss and self.for_train:
                    answer_start_idx = self._find_answer_start(taskname, input_ids, answer_field, doc)

                self.examples.append((taskname_id, input_ids, answer_start_idx))
            else:
                skipped += 1

        logging.info(f'Skipped {skipped} sentences, sequence length too short or too long even after truncation')
    '''

    def _input_sanity_checks(
        self,
        total_virtual_tokens,
        virtual_token_splits,
        prompt_template,
        prompt_template_fields,
        truncation_field,
        answer_only_loss,
        answer_field,
        doc,
    ):
        # Sanity check amount of virtual token
        assert (
            total_virtual_tokens < self.max_seq_length
        ), "virtual prompt tokens should not exceed max sequence length"

        # Make sure virtual token splits add up to the total number of virtual tokens
        assert (
            sum(virtual_token_splits) == total_virtual_tokens
        ), "Sum of prompt token split values must equal total number of prompt tokens"

        # Make sure number of virtual prompt locations match the number of virtual prompt splits
        assert prompt_template.count('<|VIRTUAL_PROMPT_') == len(
            virtual_token_splits
        ), "The number of '<|VIRTUAL_PROMPT_n|>' markers and the number of prompt token splits must match"

        # Check if input example has fields not present in template
        # keys_not_in_template = list(set(doc.keys()) - set(prompt_template_fields) - set(['taskname']))
        # assert (
        #     len(keys_not_in_template) == 0
        # ), f"Examples in your dataset contain the fields: {keys_not_in_template} that are not in the task template."

        # Answer field checks
        if answer_only_loss and self.for_train:
            assert answer_field is not None, "If answer_only_loss=True, an answer_field must be given"
            assert (
                answer_field in doc.keys()
            ), f"answer_only_loss=True but the given answer_field '{answer_field}' is not in data json"
            assert truncation_field != answer_field, "Answer field and truncation field should not match"

            answer_placeholder = "{" + answer_field + "}"
            answer_placeholder_len = len(answer_placeholder)
            placeholder_start = len(prompt_template) - answer_placeholder_len
            assert prompt_template[placeholder_start:] == answer_placeholder, "Answer field must be at prompt end"

    def _insert_text_in_template(self, input_example, prompt_template_fields, doc):
        """ Format the input example according to the template """
        for field in prompt_template_fields:
            if field in doc.keys():
                if not self.for_train and field == "answer": # during inference, do not use text in answer field
                    input_example = input_example.replace('{' + field + '}', "")
                field_text = doc[field]
                input_example = input_example.replace('{' + field + '}', field_text)

            # If some fields from the template aren't present, e.g. {answer} during inference
            # just remove that field from the template, leaving the space blank
            else:
                input_example = input_example.replace('{' + field + '}', "")

        return input_example.strip(" ")

    def _insert_virtual_token_placeholders(self, input_example, virtual_token_splits):
        """ Insert the correct number of pseudo tokens at the <|VIRTUAL_PROMPT_n|> markers """
        total_inserted_tokens = 0

        for idx in range(len(virtual_token_splits)):
            split_start = total_inserted_tokens
            split_end = total_inserted_tokens + virtual_token_splits[idx]
            pseudo_tokens_for_split = "".join(self.pseudo_tokens[split_start:split_end])
            input_example = input_example.replace(f'<|VIRTUAL_PROMPT_{idx}|>', pseudo_tokens_for_split)
            total_inserted_tokens = split_end

        return input_example

    def _truncate_input(
        self, truncation_field, input_ids, taskname, doc, prompt_template, prompt_template_fields, virtual_token_splits
    ):
        """ Try to truncate input text to fit into the max sequence length """
        logging.info(
            f"Input greater than max sequence length. Attempting to truncate: '{truncation_field}' in task: '{taskname}'"
        )

        # Truncate the text ids in this part of input to try and fit max sequence length
        if truncation_field is not None and truncation_field in doc.keys():
            truncation_length = (len(input_ids) - self.max_seq_length) + 1
            field_text = doc[truncation_field]

            # Truncate field text
            field_text_ids = self.tokenizer.text_to_ids(field_text)
            truncated_text_ids = field_text_ids[: -min(truncation_length, len(field_text_ids))]
            truncated_field_text = self.tokenizer.ids_to_text(truncated_text_ids)
            doc[truncation_field] = truncated_field_text

            # Re-insert the truncated text string into the text prompt
            input_example = prompt_template
            input_example = self._insert_text_in_template(input_example, prompt_template_fields, doc)
            input_example = self._insert_virtual_token_placeholders(input_example, virtual_token_splits)

            # Re-tokenize the whole prompt
            input_ids = self.tokenizer.text_to_ids(input_example)

            # not sure
            # Add BOS/EOS if desired, adds EOS by default
            if self.add_bos:
                input_ids = [self.tokenizer.bos_id] + input_ids
            if self.add_eos:
                input_ids = input_ids + [self.tokenizer.eos_id]

            # pad the question so 'answer:' coincides with the end of the first chunk of 64
            answer_field = self.task_templates[taskname]["answer_field"]

            answer_only_loss = self.task_templates[taskname]["answer_only_loss"]
            answer_start_idx = len(input_ids)
            if answer_only_loss and self.for_train:
                answer_start_idx = self._find_answer_start(taskname, input_ids, answer_field, doc)
            length_before_answer = answer_start_idx
            # todo: consider virtual prompt length and make the whole sequence has a length of a mutiple of chunk size
            if length_before_answer < 64:
                padding_length = 64 - length_before_answer
                input_ids = [self.pad_token_id] * padding_length + input_ids
                
                
            
        return input_ids

    def _find_answer_start(self, taskname, input_ids, answer_field, doc):
        """ Find the token ids corresponding to the answer start, for loss masking purposes.
            Assumes the answer is always at the end of the prompt.
        """
        answer_text = doc[answer_field]
        answer_text = self._add_leading_space(taskname, answer_field, answer_text)
        answer_text_ids = self.tokenizer.text_to_ids(answer_text)
        num_answer_text_ids = len(answer_text_ids)

        if self.add_eos:
            num_answer_text_ids += 1

        answer_start_idx = len(input_ids) - num_answer_text_ids

        return answer_start_idx

    def _add_leading_space(self, taskname, field_name, field_text):
        """ Add leading space to text if there is a space before it in the template """
        prompt_template = self.task_templates[taskname]["prompt_template"]
        field_text_start = prompt_template.find("{" + field_name + "}")
        if field_text_start != 0 and prompt_template[field_text_start - 1] == " ":
            field_text = " " + field_text

        return field_text

    def __len__(self):
        """Inherit __len__ function from RetroQAFineTuneDataset"""
        return super().__len__()

    def __getitem__(self, idx):
        """Inherit __getitem__ function from RetroQAFineTuneDataset"""
        return super().__getitem__(idx)

    def collate_fn(self, batch, tp_workers=0):
        """ Prepares input_ids, labels, loss mask, attention_mask, and position ids for global batch """
        # input_ids, answer_starts, chunks, num_virtual_tokens = zip(*batch)
        # pad strategy 3
        input_ids, answer_starts, chunks, num_virtual_tokens = zip(*batch)
        num_virtual_tokens = num_virtual_tokens[0]  # all items in the list of num_virtual_tokens should be the same
        # convert chunks into torch tensors
        chunks = list(chunks)
        max_num_chunks = max(c.shape[0] for c in chunks)
        padding_chunks = np.full((1, chunks[0].shape[1], chunks[0].shape[2]), self.pad_token_id) # (1, num_neighbor, num_token_in_neighbor)
        for i in range(len(chunks)):
            if chunks[i].shape[0] < max_num_chunks:
                chunks[i] = np.concatenate((chunks[i], np.repeat(padding_chunks, max_num_chunks-chunks[i].shape[0], axis=0)), axis=0)
        chunks = torch.tensor(chunks)

        # Get max sequence length of batch
        batch_max = max(len(ids) for ids in input_ids)

        if tp_workers > 1:
            # make sure the sequence length is multiply of number of tp_workers, needed for sequence parallel.
            resi_padding = (tp_workers - (batch_max - 1) % tp_workers) % tp_workers
        else:
            resi_padding = 0
        batch_max += resi_padding
        input_ids, loss_mask = self.pad_batch_and_build_loss_mask(input_ids, batch_max, answer_starts)
        # Should be a label for every token in batch, label is the next token
        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        batch_max -= 1

        # Loss mask should align with labels
        loss_mask = loss_mask[:, 1:].contiguous()

        if self.megatron_lm_compatible: # megatron lm retro model does not use masks
            hidden_mask = torch.ones_like(input_ids, dtype=torch.bool)
            context_mask = torch.ones_like(chunks, dtype=torch.bool)
        else:
            hidden_mask = input_ids != self.pad_token_id
            context_mask = chunks != self.pad_token_id
        
      

        # lob off the space holder for virtual tokens
        # input_ids = input_ids[:, num_virtual_tokens:]

        # pad strategy 3: do not lob off here

        # input_ids: eos padding + vt + real tokens + batch padding
        return input_ids, hidden_mask, loss_mask, chunks, context_mask, labels

    def pad_batch_and_build_loss_mask(self, input_ids, batch_max, answer_starts):
        """ Pad input_ids in batch to max batch length while building loss mask """
        batch_loss_masks = []
        padded_input_ids = []
        for ids, answer_start_idx in zip(input_ids, answer_starts):
            if self.answer_only_loss and answer_start_idx is not None:
                # Loss mask where answer tokens are 1.0 and all other tokens are 0.0
                loss_mask = [float(idx >= answer_start_idx) for idx in range(len(ids))]
            else:
                # Loss mask where virtual tokens are 0.0 and all other tokens are 1.0
                loss_mask = [float(token_id not in self.pseudo_token_ids) for token_id in ids]
            # Pad to max length
            input_length = len(ids)
            padding_length = batch_max - input_length
            ids = ids + [self.pad_token_id] * padding_length
            padded_input_ids.append(ids)

            # Account for padding in loss mask
            loss_mask.extend([0.0] * padding_length)
            batch_loss_masks.append(torch.tensor(loss_mask, dtype=torch.float))

        # Make into torch tensors
        padded_input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
        batch_loss_masks = torch.stack(batch_loss_masks)

        return padded_input_ids, batch_loss_masks

    def inference_collate_fn(self, batch, tp_workers=0):
        """
        Used for loading inference data. 
        """
        input_ids, answer_starts, chunks, num_virtual_tokens = zip(*batch)
        num_virtual_tokens = num_virtual_tokens[0]  # all items in the list of num_virtual_tokens should be the same

        chunks = list(chunks)
        chunks = [c[0] for c in chunks]

        # # Get max sequence length of batch
        # batch_max = max(len(ids) for ids in input_ids)

        # if tp_workers > 1:
        #     # make sure the sequence length is multiply of number of tp_workers, needed for sequence parallel.
        #     resi_padding = (tp_workers - (batch_max - 1) % tp_workers) % tp_workers
        # else:
        #     resi_padding = 0
        # batch_max += resi_padding
        # input_ids, loss_mask = self.pad_batch_and_build_loss_mask(input_ids, batch_max, answer_starts)

        # lob off the space holder for virtual tokens
        
        # input_ids = [ids[num_virtual_tokens:] for ids in input_ids]
        
        return (input_ids, chunks)

