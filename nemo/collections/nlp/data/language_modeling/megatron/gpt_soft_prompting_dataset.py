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
        dataset_paths,
        tokenizer,
        soft_token_source: str,
        task_templates: dict,
        total_soft_tokens: int,
        pseudo_token: str,
        pad_token_id: str,
        max_seq_length: int,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
    ):
        self.tokenizer = tokenizer
        self.soft_token_source = soft_token_source
        self.task_templates = task_templates
        self.total_soft_tokens = total_soft_tokens
        self.pseudo_token = pseudo_token
        self.pseudo_token_id = self.tokenizer.token_to_id(pseudo_token)
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.examples = []

        assert min_seq_length <= max_seq_length, "Min sequence length should be less than or equal to max"
        assert max_seq_length > 0, "Max sequence length should be greater than 0"
        assert total_soft_tokens > 0, "There should be at least one soft prompt token"
        assert total_soft_tokens < max_seq_length, "Soft prompt tokens should not exceed max sequence length"

        logging.info("Loading and tokenizing dataset ... ")

        for path in dataset_paths:
            dataset_file = open(path, 'r', encoding='utf-8')
            skipped = 0

            for json_line in tqdm(dataset_file):

                # Load the information for a single example from .json file
                doc = json.loads(json_line)
                taskname = doc["taskname"]
                prompt_template = self.task_templates[taskname]["prompt_template"]
                prompt_token_splits = self.task_templates[taskname]["prompt_token_splits"]
                input_example = prompt_template

                assert sum(prompt_token_splits) == self.total_soft_tokens, "Sum of prompt token splits must equal total number of prompt tokens"
                assert prompt_template.count('<|SOFT_PROMPT_') == len(prompt_token_splits), "The number of '<|SOFT_PROMPT_n|>' markers and the number of prompt token splits must match"

                # Format the input example according to the template
                for field in doc.keys():
                    field_text = doc[field]
                    input_example = input_example.replace('{' + field + '}', field_text)

                # Insert the correct number of pseudo tokens at the <|SOFT_PROMPT_n|> markers
                for idx in range(len(prompt_token_splits)):
                    input_example = input_example.replace(f'<|SOFT_PROMPT_{idx}|>', self.pseudo_token * prompt_token_splits[idx])
                input_ids = self.tokenizer.text_to_ids(input_example)

                # Add BOS/EOS if desired, adds EOS by default
                if self.add_bos:
                    input_ids = [self.tokenizer.bos_id] + input_ids
                if self.add_eos:
                    input_ids = input_ids + [self.tokenizer.eos_id]

                # Skip example if the final length doesn't fit length requirements
                if self.min_seq_length <= len(input_ids) <= self.max_seq_length:
                    if self.soft_token_source == "prompt-encoder":
                        taskname_id = self.tokenizer.text_to_ids(taskname)

                    elif self.soft_token_source == "prompt-table":
                        taskname_id = self.task_templates[taskname]["task_id_num"]

                    self.examples.append((taskname_id, input_ids))

                else:
                    skipped += 1

            logging.info(f'Skipped {skipped} sentences, sequence length too long or too short in dataset {path}')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def collate_fn(self, batch):
        """ Prepares input_ids, labels, loss mask, attention_mask, and position ids for global batch """
        # Get max sequence length of batch
        taskname_ids, input_ids = zip(*batch)

        # Pad taskname_ids to be the same length for the prompt encoder
        if self.soft_token_source == "prompt-encoder":
            max_taskname_length = max(len(ids) for ids in taskname_ids)
            taskname_ids = [ids + [self.pad_token_id] * (max_taskname_length - len(ids)) for ids in taskname_ids]
            taskname_ids = torch.tensor(taskname_ids)
            
        # Task ids are just used for a look up embeddings for prompt-table
        elif self.soft_token_source == "prompt-table":
            taskname_ids = torch.tensor(taskname_ids)

        batch_max = max(len(ids) for ids in input_ids)
        input_ids, loss_mask = self.pad_batch_and_build_loss_mask(input_ids, batch_max)

        # Should be a label for every token in batch, label is the next token
        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        batch_max -= 1

        # Loss mask should align with labels
        loss_mask = loss_mask[:, 1:].contiguous()

        # Using causal attention mask for whole input
        batch_size = len(input_ids)
        attention_mask = torch.tril(torch.ones((batch_size, batch_max, batch_max))).view(
            batch_size, 1, batch_max, batch_max
        )

        # Convert attention mask from float to bool
        attention_mask = attention_mask < 0.5
        position_ids = build_position_ids(input_ids)

        return input_ids, labels, loss_mask, attention_mask, position_ids, taskname_ids

    def pad_batch_and_build_loss_mask(self, input_ids, batch_max):
        """ Pad input_ids in batch to max batch length while building loss mask """
        batch_loss_masks = []
        for ids in input_ids:
            # Loss mask where soft tokens are 0.0 and all other tokens are 1.0
            loss_mask = np.where(np.array(ids) == self.pseudo_token_id, 0.0, 1.0)
            loss_mask = list(loss_mask)

            # Pad to max length
            input_length = len(ids)
            padding_length = batch_max - input_length
            ids.extend([self.pad_token_id] * padding_length)

            # Account for padding in loss mask
            loss_mask.extend([0.0] * padding_length)
            batch_loss_masks.append(torch.tensor(loss_mask, dtype=torch.float))

        # Make into torch tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        batch_loss_masks = torch.stack(batch_loss_masks)

        return input_ids, batch_loss_masks