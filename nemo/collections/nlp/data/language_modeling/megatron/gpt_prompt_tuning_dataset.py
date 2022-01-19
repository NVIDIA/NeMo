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

"""
Prompt tuning dataset
Expects data to be in the format:
{"prompt_tag": "tag1", "text": "example question1", "answer": "answer1"}
{"prompt_tag": "tag1", "text": "example question2", "answer": "answer2"}
{"prompt_tag": "tag2", "text": "example question3", "answer": "answer3"}

"""
import json

import torch
from tqdm import tqdm

from nemo.core import Dataset
from nemo.utils import logging

__all__ = ["GPTPromptTuningDataset"]


class GPTPromptTuningDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        tokenizer,
        num_prompt_tokens: int,
        max_seq_length: int,
        min_seq_length: int = 1,
        add_bos_eos: bool = True,
        calc_loss_on_answer_only=True,
    ):
        self.tokenizer = tokenizer
        self.add_bos_eos = add_bos_eos
        self.calc_loss_on_answer_only = calc_loss_on_answer_only
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.num_prompt_tokens = num_prompt_tokens
        self.max_sent_length = max_seq_length - num_prompt_tokens
        self.tags_and_tokens = []

        assert min_seq_length <= max_seq_length, "Min sequence length should be less than or equal to max"
        assert max_seq_length > 0, "Max sequence length should be greater than 0"
        assert num_prompt_tokens > 0, "There should be at least one soft prompt token"
        assert num_prompt_tokens < max_seq_length, "Soft prompt tokens should not exceed max sequence length"

        dataset_file = open(dataset_path, 'r', encoding='utf-8')

        logging.info("Loading and tokenizing dataset ... ")

        skipped = 0
        for json_line in tqdm(dataset_file):
            doc = json.loads(json_line)
            prompt_tag = doc["prompt_tag"]
            question = str(doc["text"])
            answer = str(doc["answer"])  # Incase 'True' or 'False' gets read as bool
            sent = question + answer

            sent_ids = tokenizer.text_to_ids(sent)
            answer_ids = tokenizer.text_to_ids(answer)
            answer_len = len(answer_ids)

            if self.add_bos_eos:
                sent_ids = [tokenizer.bos_id] + sent_ids + [tokenizer.eos_id]
                answer_len += 1  # To account for EOS token

            # Need to leave space for prompt tokens in sequence
            if self.min_seq_length <= len(sent_ids) <= self.max_sent_length:
                self.tags_and_tokens.append((prompt_tag, sent_ids, answer_len))

            else:
                skipped += 1

        logging.info(f'Skipped {skipped} sentences, sequence length too long or too short')

    def __len__(self):
        return len(self.tags_and_tokens)

    def __getitem__(self, idx):
        return self.tags_and_tokens[idx]

    def collate_fn(self, batch):
        """Build masks and position id for left to right model with prompt tuning."""

        prompt_tags, input_ids, answer_lens = zip(*batch)

        # Get max sequence length of batch
        batch_size = len(input_ids)
        batch_max = max(len(ids) for ids in input_ids)

        # Add prompt token length
        batch_max_with_prompt = batch_max + self.num_prompt_tokens

        # Pad tokens in batch to max batch length while building loss mask
        loss_masks = []
        for idx, ids in enumerate(input_ids):
            text_length = len(ids)
            answer_length = answer_lens[idx]

            prompt_loss_mask = [0.0] * self.num_prompt_tokens

            # Loss mask everything except the answer
            if self.calc_loss_on_answer_only:
                question_loss_mask = [0.0] * (text_length - answer_length)
                answer_loss_mask = [1.0] * answer_length
                text_loss_mask = prompt_loss_mask + question_loss_mask + answer_loss_mask

            # Loss mask soft prompt and padding only, calc loss on all text after soft prompt
            else:
                text_loss_mask = [1.0] * text_length
                text_loss_mask = prompt_loss_mask + text_loss_mask

            padding_length = batch_max - text_length

            # Pad loss mask and text tokens
            ids.extend([self.tokenizer.eos_id] * padding_length)
            text_loss_mask.extend([0.0] * padding_length)
            loss_masks.append(torch.tensor(text_loss_mask, dtype=torch.float))

        tokens = torch.tensor(input_ids, dtype=torch.long)
        loss_mask = torch.stack(loss_masks)

        # Position ids for text
        text_position_ids = torch.arange(start=self.num_prompt_tokens, end=batch_max_with_prompt, dtype=torch.long,)
        text_position_ids = text_position_ids.unsqueeze(0).expand_as(tokens).clone()

        # Attention mask (lower triangular) starting with prompt tokens
        attention_mask = torch.tril(torch.ones((batch_size, batch_max_with_prompt, batch_max_with_prompt))).view(
            batch_size, 1, batch_max_with_prompt, batch_max_with_prompt
        )

        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5

        # Labels for prompt tokens
        prompt_token_labels = torch.full(
            size=(batch_size, self.num_prompt_tokens), fill_value=self.tokenizer.bos_id, dtype=torch.long,
        )

        # Should be a label for every token in batch
        labels = torch.cat((prompt_token_labels, tokens[:, 1:].contiguous()), dim=1)
        final_label = torch.full(size=(batch_size, 1), fill_value=self.tokenizer.eos_id, dtype=torch.long,)

        # Last label should be eos, even for longest sequence in batch
        labels = torch.cat((labels, final_label), dim=1)

        return tokens, labels, prompt_tags, attention_mask, loss_mask, text_position_ids
