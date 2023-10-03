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

import array
import pickle as pkl
from typing import Optional

import torch

from nemo.collections.nlp.data.data_utils.data_preprocessing import find_newlines, load_data_indices
from nemo.core.classes import Dataset
from nemo.utils import logging

__all__ = ['EntityLinkingDataset']


class EntityLinkingDataset(Dataset):
    """
    Parent class for entity linking encoder training and index
    datasets

    Args:
        tokenizer (obj): huggingface tokenizer,
        data_file (str): path to tab separated column file where data 
            pairs apear in the format 
            concept_ID\tconcept_synonym1\tconcept_synonym2\n
        newline_idx_file (str): path to pickle file containing location
            of data_file newline characters
        max_seq_length (int): maximum length of a concept in tokens
        is_index_data (bool): Whether dataset will be used for building
                            a nearest neighbors index
    """

    def __init__(
        self,
        tokenizer: object,
        data_file: str,
        newline_idx_file: Optional[str] = None,
        max_seq_length: Optional[int] = 512,
        is_index_data: bool = False,
    ):

        self.tokenizer = tokenizer

        # Try and load pair indices file if already exists
        newline_indices, newline_idx_file, _ = load_data_indices(newline_idx_file, data_file, "newline_indices")

        # If pair indices file doesn't exists, generate and store them
        if newline_indices is None:
            logging.info("Getting datafile newline indices")

            with open(data_file, "rb") as f:
                contents = f.read()
                newline_indices = find_newlines(contents)
                newline_indices = array.array("I", newline_indices)

            # Store data file indicies to avoid generating them again
            with open(newline_idx_file, "wb") as f:
                pkl.dump(newline_indices, f)

        self.newline_indices = newline_indices
        self.data_file = data_file
        self.num_lines = len(newline_indices)
        self.max_seq_length = max_seq_length
        self.is_index_data = is_index_data

        logging.info(f"Loaded dataset with {self.num_lines} examples")

    def __len__(self):
        return self.num_lines

    def __getitem__(self, idx):

        concept_offset = self.newline_indices[idx]

        with open(self.data_file, "r", encoding='utf-8-sig') as f:
            # Find data pair within datafile using byte offset
            f.seek(concept_offset)
            concept = f.readline()[:-1]
            concept = concept.strip().split("\t")

            if self.is_index_data:
                concept_id, concept = concept
                return (int(concept_id), concept)

            else:
                concept_id, concept1, concept2 = concept
                return (int(concept_id), concept1, concept2)

    def _collate_fn(self, batch):
        """collate batch of input_ids, segment_ids, input_mask, and label

        Args:
            batch:  A list of tuples of format (concept_ID, concept_synonym1, concept_synonym2).
        """
        if self.is_index_data:
            concept_ids, concepts = zip(*batch)
            concept_ids = list(concept_ids)
            concepts = list(concepts)

        else:
            concept_ids, concepts1, concepts2 = zip(*batch)
            concept_ids = list(concept_ids)
            concept_ids.extend(concept_ids)  # Need to double label list to match each concept
            concepts = list(concepts1)
            concepts.extend(concepts2)

        batch = self.tokenizer(
            concepts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_length=True,
        )

        return (
            torch.LongTensor(batch["input_ids"]),
            torch.LongTensor(batch["token_type_ids"]),
            torch.LongTensor(batch["attention_mask"]),
            torch.LongTensor(concept_ids),
        )
