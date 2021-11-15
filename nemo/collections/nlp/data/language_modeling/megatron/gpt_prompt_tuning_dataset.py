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

"""Prompt tuning dataset"""
import json

from tqdm import tqdm

from nemo.core import Dataset
from nemo.utils import logging


__all__ = ["PromptTuningDataset"]


class GPTPromptTuningDataset(Dataset):
    def __init__(
        self, 
        dataset_path: str,
        tokenizer: Any,
        num_prompt_tokens: int,
        max_seq_length: int,
        min_seq_length: int = 1,
        add_bos_eos: bool = True,

    ):
        self.tokenizer = tokenizer
        self.add_bos_eos = add_bos_eos
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.num_prompt_tokens = num_prompt_tokens
        self.max_sent_length = max_seq_length - num_prompt_tokens
        self.prompt_tags = []
        self.input_ids = []

        dataset_file = open(dataset_path, 'r', encoding='utf-8')

        logging.info("Loading and tokenizing dataset ... ")

        skipped = 0
        for json_line in tqdm(dataset_file):
            doc = json.loads(json_line)
            sent = doc["text"]
            prompt_tag = doc["prompt_tag"]

            sent_ids = tokenizer.text_to_ids(sent.decode('utf-8'))

            if self.add_bos_eos:
                sent_ids = [tokenizer.bos_id] + sent_ids + [tokenizer.eos_id]

            # Need to leave space for prompt tokens in sequence
            if self.min_seq_length <= len(sent_ids) <= self.max_sent_length:
                self.prompt_tags.append(prompt_tag)
                self.input_ids.append(sent_ids)

            else:
                skipped += 1

            logging.info(f'Skipped {skipped} sentences, sequence length too long or too short') 


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        prompt_tags = self.prompt_tags[idx]
        input_ids = self.input_ids[idx] 
        labels = input_ids[1:].copy() + [self.tokenizer.eos_id]

        return prompt_tag, input_ids

    def collate_fn(self, batch):
        """Build masks and position id for left to right model with prompt tuning."""

        prompt_tags, input_ids = zip(*batch)

        # Get max sequence length of batch
        batch_size = len(input_ids)
        batch_max = max(len(ids) for ids in input_ids)

        # Add prompt token length
        batch_max_with_prompt = batch_max + self.num_prompt_tokens

        # Pad tokens in batch to max batch length while building loss mask
        loss_masks = []
        for ids in input_ids:
            text_length = len(ids)

            # Loss mask starting with prompt tokens
            text_loss_mask = [1.0] * (self.num_prompt_tokens + text_length)
            padding_length = batch_max - text_length

            # Pad loss mask and text tokens
            ids.extend([self.tokenizer.eod_id] * padding_length)
            text_loss_mask.extend([0.0] * padding_length)
            loss_masks.append(text_loss_mask)

        tokens = torch.tensor(input_ids, dtype=torch.long, device=data.device)
        loss_mask = torch.stack(loss_masks, dtype=torch.float, device=data.device)

        # Position ids for prompts
        prompt_position_ids = torch.arange(self.num_prompt_tokens, dtype=torch.long, device=data.device)
        prompt_position_ids = prompt_position_ids.unsqueeze(0).expand((batch_size, self.num_prompt_tokens))

        # Position ids for text
        text_position_ids = torch.arange(start=self.num_prompt_tokens, end=batch_max, dtype=torch.long, device=data.device)
        text_position_ids = text_position_ids.unsqueeze(0).expand_as(tokens)

        # Attention mask (lower triangular) starting with prompt tokens
        attention_mask = torch.tril(torch.ones((batch_size, batch_max_with_prompt, batch_max_with_prompt), device=data.device)).view(
            batch_size, 1, seq_length, seq_length
        )

        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5
        
        # Should be a label for every token in batch
        labels = tokens[:, 1:].contiguous()
        final_label = torch.constant(self.tokenizer.eos_id, 
                                     dtype=torch.long, 
                                     shape=(batch_size, 1), 
                                     device=data.device,
                                     )

        # Last label should be eos, even for longest sequence in batch
        labels = torch.cat((labels, final_label), dim=1)

        return tokens, lables, prompt_tags, attention_mask, loss_mask, prompt_position_ids, text_position_ids

