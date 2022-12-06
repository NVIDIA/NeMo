# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.nlp.modules.common.megatron.utils import build_position_ids
from nemo.core import Dataset
from nemo.utils import AppState, logging

__all__ = ['GPTSupervisedFineTuningDataset']


class GPTSupervisedFineTuningDataset(Dataset):
    """
    The dataset class for performing supervised fine-tuning on GPT models
    
    Args:
        data (list[strings], list[dicts]): (1) paths to .jsonl or .json files, (2) dict objects corresponding to each input example
        tokenizer (tokenizer): Tokenizer from frozen language model
        pad_token_id (int): ID of pad token from tokenizer
        max_seq_length (int): maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset. Data examples will be dropped if they do not meet the min length requirements. 
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add an end of sentence token to each data example
        add_sep (bool): Whether to add a separation token to each data example (goes between prompt and answer)
        for_train (bool): Whether you're creating a dataset for training or inference
        tokens_to_generate (int): (inference only) Number of tokens to generate during inference
        cache_data_path (str): The path to the cached dataset
        load_cache (bool): Whether to load cached dataset
    """

    def __init__(
        self,
        data,
        tokenizer,
        pad_token_id: int,
        max_seq_length: int,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        add_sep: bool = True,
        sep_id: int = None,
        for_train: bool = True,
        answer_only_loss = True,
        tokens_to_generate=None,
        cache_data_path: str = None,  # the cache file
        load_cache: bool = True,  # whether to load from the cache if it is available
    ):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.add_sep = add_sep
        self.sep_id = sep_id
        self.answer_only_loss = answer_only_loss
        self.for_train = for_train
        self.examples = []

        if self.sep_id is None:
            self.sep_id = self.tokenizer.sep_id

        if not self.for_train:
            self.tokens_to_generate = tokens_to_generate

        assert self.min_seq_length <= max_seq_length, "Min sequence length should be less than or equal to max"
        assert self.max_seq_length > 0, "Max sequence length should be greater than 0"

        logging.info("Loading and tokenizing dataset ... ")

        if load_cache and cache_data_path is not None and os.path.exists(cache_data_path):
            # load it from the cache
            logging.info(f'load the data from the cache file {cache_data_path}')
            with open(cache_data_path, 'rb') as f:
                self.examples = pickle.load(f)
        else:
            # Data is just a list of dicts already loaded from a json file or passed in directly as a dict
            if isinstance(data[0], dict):
                self.load_data(data)

            # Datasets are a list of file path strings to .json or .jsonl files
            elif isinstance(data[0], str):
                for path in data:
                    dataset = open(path, 'r', encoding='utf-8')
                    self.load_data(dataset)
            else:
                raise ValueError("Datasets must be a list of filepath strings or a list of data example dicts")
            if cache_data_path is not None:
                # the first worker save the results into the cache file
                app_state = AppState()
                if app_state._global_rank == 0:
                    with open(cache_data_path, 'wb') as f:
                        pickle.dump(self.examples, f)
                    logging.info(f'save the data to the cache file {cache_data_path}')

    def load_data(self, dataset):
        """
        Loads a dataset by concatenating text and answer fields. Adds BOS/SEP/EOS tokens when
        required. Converts all input text into token ids. 

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

            self._input_sanity_checks(
                doc,
            )

            # Format and truncate inputs (text + sep + answer)
            input_ids, answer_start_idx = self._append_text_answer_and_truncate(doc)

            # Add BOS/EOS if desired, adds EOS by default
            if self.add_bos:
                input_ids = [self.tokenizer.bos_id] + input_ids
            if self.add_eos:
                input_ids = input_ids + [self.tokenizer.eos_id]

            # Skip example if the final length doesn't fit length requirements even after truncation
            if self.min_seq_length <= len(input_ids) <= self.max_seq_length:
                self.examples.append((input_ids, answer_start_idx))
            else:
                skipped += 1

        logging.info(f'Skipped {skipped} sentences, sequence length too short or too long even after truncation')

    def _input_sanity_checks(
        self,
        doc,
    ):
        # Sanity check amount of virtual token

        # Answer field checks
        if self.for_train:
            assert (
                'answer' in doc.keys()
            ), f"You must provide both 'text' and 'answer' fields"
            # Truncation field will always be answer

    def _append_text_answer_and_truncate(self, doc):
        """ Format the input example according to the template """

        text_ids = self.tokenizer.text_to_ids(doc['text'])
        answer_ids = self.tokenizer.text_to_ids(doc['answer'])

        total_ids = len(text_ids) + len(answer_ids)
        if self.add_bos:
            total_ids += 1
        if self.add_sep:
            total_ids += 1
        if self.add_eos:
            total_ids +=1

        # If the total number of token is greater than the max, we will try to truncate the answer
        if total_ids > self.max_seq_length:
            truncation_length = total_ids - self.max_seq_length
            answer_ids = answer_ids[:-min(truncation_length, len(answer_ids))]

        input_ids = text_ids

        # Adds sep token between text/prompt and answer
        if self.add_sep:
            input_ids = input_ids + [self.sep_id]
        
        answer_start_idx = len(input_ids)

        input_ids = input_ids + answer_ids

        return input_ids, answer_start_idx

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def collate_fn(self, batch, tp_workers=0):
        """ Prepares input_ids, labels, loss mask, attention_mask, and position ids for global batch """
        input_ids, answer_starts = zip(*batch)

        # Get max sequence length of batch
        batch_max = max(len(ids) for ids in input_ids)

        if tp_workers > 1:
            # more sure the sequence length is multiple of number of tp_workers, needed for sequence parallel.
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

        # Using causal attention mask for whole input
        batch_size = len(input_ids)
        attention_mask = torch.tril(torch.ones((batch_size, batch_max, batch_max))).view(
            batch_size, 1, batch_max, batch_max
        )

        # Convert attention mask from float to bool
        attention_mask = attention_mask < 0.5
        position_ids = build_position_ids(input_ids)

        return input_ids, labels, loss_mask, attention_mask, position_ids

    def pad_batch_and_build_loss_mask(self, input_ids, batch_max, answer_starts):
        """ Pad input_ids in batch to max batch length while building loss mask """
        batch_loss_masks = []
        padded_input_ids = []
        for ids, answer_start_idx in zip(input_ids, answer_starts):
            if answer_start_idx is not None:
                # Loss mask where answer tokens are 1.0 and all other tokens are 0.0
                loss_mask = [float(idx >= answer_start_idx) for idx in range(len(ids))]
            # TODO: @fsoares to double-check all sequence loss. Args are already in class

            # Pad to max length
            input_length = len(ids)
            padding_length = batch_max - input_length
            pad_extend = [self.pad_token_id] * padding_length
            ids = ids + pad_extend
            padded_input_ids.append(ids)

            # Account for padding in loss mask
            loss_mask.extend([0.0] * padding_length)
            batch_loss_masks.append(torch.tensor(loss_mask, dtype=torch.float))

        # Make into torch tensors
        padded_input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
        batch_loss_masks = torch.stack(batch_loss_masks)

        return padded_input_ids, batch_loss_masks

    def inference_collate_fn(self, batch):
        """
        Used for loading inference data. 
        """
        input_ids, answer_starts = zip(*batch)
        input_lengths = torch.cuda.LongTensor([len(inputs) for inputs in input_ids])
        #task_id_nums = torch.cuda.LongTensor(task_id_nums)
        batch_max = input_lengths.max().item()
        batch_max += self.tokens_to_generate

        input_ids, _ = self.pad_batch_and_build_loss_mask(input_ids, batch_max, answer_starts)
        input_ids = input_ids.cuda()
        input_ids = torch.cuda.LongTensor(input_ids)

        return (input_ids, input_lengths)

