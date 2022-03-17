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

#TODO: Move this to documentation
"""
Prompt tuning/p-tuning dataset

Expects data json to have at least 5 fields: 
    taskname
    prompt_template
    prompt_token_splits
    answer
    [+ unlimted number of other fields, must have at least 1]

taskname is the name of the task

The template should have the special charaters "<|SOFT_PROMPT|>" 
anywhere the user wants virtual tokens to be inserted. 

prompt_token_splits specifies how many virtual tokens to insert
at each <|SOFT_PROMPT|> location. The total number of virtual
tokens inserted into your input must add up to the total_prompt_tokens
value specified by the user in the config. 

The other fields in the json that are not in the first four listed 
above should match the ones specified in the prompt_template. 

An example where you want to prepend all virtual tokens to the front
of your text input might look like:

{
    "taskname": "boolq", 
    "prompt_template": "<|SOFT_PROMPT|> Question: {question} Answer: {answer}",
    "prompt_token_splits": [100],
    "question":
    "answer": 
}

An example where you want to insert virtual tokens throughout your 
input might look like:
{
    "taskname": "boolq",
    "prompt_template": "<|SOFT_PROMPT|> Question: {question} <|SOFT_PROMPT|> Answer: {answer}",
    "prompt_token_splits": [60, 40],
    "question": 
    "answer": 
}

another example with soft prompts being insert at 4 locations: 

{
    "taskname": "intent_slot_pred",
    "prompt_template": "<|SOFT_PROMPT|> Utterance: {utterance} <|SOFT_PROMPT|> intent {slots} <|SOFT_PROMPT|>
}

"""
import json
import numpy as np
import torch
from tqdm import tqdm

from nemo.collections.nlp.modules.common.megatron.utils import build_position_ids
from nemo.core import Dataset
from nemo.utils import logging

__all__ = ["GPTSoftPromptDataset"]


class GPTSoftPromptDataset(Dataset):
    """
    The dataset class for prompt-tuning or p-tuning GPT models.
    """
    def __init__(
        self,
        dataset_path,
        tokenizer,
        prompt_templates: dict,
        total_prompt_tokens: int,
        pseudo_token: str,
        taskname_to_id: dict,
        max_seq_length: int,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        calc_loss_on_answer_only=False,
    ):
        self.tokenizer = tokenizer
        self.prompt_templates = prompt_templates
        self.total_prompt_tokens = total_prompt_tokens
        self.pseudo_token = pseudo_token
        self.pseudo_token_id = self.tokenizer.token_to_id(pseudo_token)
        self.taskname_to_id = taskname_to_id
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.calc_loss_on_answer_only = calc_loss_on_answer_only
        self.examples = []

        assert min_seq_length <= max_seq_length, "Min sequence length should be less than or equal to max"
        assert max_seq_length > 0, "Max sequence length should be greater than 0"
        assert num_prompt_tokens > 0, "There should be at least one soft prompt token"
        assert num_prompt_tokens < max_seq_length, "Soft prompt tokens should not exceed max sequence length"

        logging.info("Loading and tokenizing dataset ... ")
        dataset_file = open(dataset_path, 'r', encoding='utf-8')
        skipped = 0

        for json_line in tqdm(dataset_file):

            # Load the information for a single example from .json file
            doc = json.loads(json_line)
            taskname = doc["taskname"]
            prompt_template = prompt_templates[taskname]["prompt_template"]
            prompt_token_splits = prompt_templates[taskname]["prompt_token_splits"]
            input_example = prompt_template

            assert sum(prompt_token_splits) == self.total_prompt_tokens, "Sum of prompt token splits must equal total number of prompt tokens"
            assert prompt_template.count('<|SOFT_PROMPT_') == len(prompt_token_splits), "The number of '<|SOFT_PROMPT_n|>' markers and the number of prompt token splits must match"

            # Format the input example according to the template
            for field in doc.keys():
                field_text = doc[field]
                input_example = input_example.replace('{' + field + '}', field_text)

            # Insert the correct number of pseudo tokens at the <|SOFT_PROMPT_n|> markers
            for idx in range(len(prompt_token_splits)):
                input_example = input_example.replace(f'<|SOFT_PROMPT_{idx}|>', self.pseudo_token * prompt_token_splits[idx])
            input_ids = tokenizer.text_to_ids(input_example)

            # Add BOS/EOS if desired, adds EOS by default
            if self.add_bos:
                input_ids = [tokenizer.bos_id] + input_ids
            if self.add_eos:
                input_ids = input_ids + [tokenizer.eos_id]

            # Skip example if the final length doesn't fit length requirements
            if self.min_seq_length <= len(input_ids) <= self.max_seq_length:
                taskname_id = self.taskname_to_id[taskname]
                self.examples.append((taskname_id, input_ids))
            else:
                skipped += 1

        logging.info(f'Skipped {skippped} sentences, sequence length too long or too short')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def collate_fn(self, batch):
        """ Prepares a batch for all soft prompting methods"""

        taskname_ids, input_ids = zip(*batch)
        taskname_ids = torch.tensor(taskname_ids)
        input_ids, labels, loss_mask, attention_mask, position_ids = self.process_global_batch(input_ids)

        return input_ids, labels, loss_mask, attention_mask, position_ids, taskname_ids

    def process_global_batch(self, input_ids):
        """ Prepares input_ids, labels, loss mask, attention_mask, and position ids for global batch """
        # Get max sequence length of batch
        batch_size = len(input_ids)
        batch_max = max(len(ids) for ids in input_ids)
        input_ids, loss_mask = self.pad_batch_and_build_loss_mask(input_ids, batch_max)

        # Should be a label for every token in batch, label is the next token
        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()

        # Loss mask should align with labels
        loss_mask = loss_mask[:, 1:]

        # Using causal attention mask for whole input
        attention_mask = torch.tril(torch.ones((batch_size, batch_max, batch_max))).view(
            batch_size, 1, batch_max, batch_max
        )

        # Convert attention mask from float to bool
        attention_mask = attention_mask < 0.5
        position_ids = build_position_ids(input_ids)

        return input_ids, labels, loss_mask, attention_mask, position_ids

    def pad_batch_and_build_loss_mask(self, input_ids, batch_max):
        """ Pad input_ids in batch to max batch length while building loss mask """
        loss_masks = []
        for idx, ids in enumerate(input_ids):
            # Loss mask where soft tokens are 0.0 and all other tokens are 1.0
            loss_mask = np.where(np.array(ids) == self.pseudo_token_id, 0.0, 1.0)
            loss_mask = list(loss_mask)

            # Pad to max length
            input_length = len(ids)
            padding_length = batch_max - input_length
            ids.extend([self.tokenizer.pad_id] * padding_length)

            # Account for padding in loss mask
            loss_mask.extend([0.0] * padding_length)
            loss_masks.append(torch.tensor(loss_mask, dtype=torch.float))

        # Make into torch tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        loss_mask = torch.stack(loss_mask)

        return input_ids, loss_masks