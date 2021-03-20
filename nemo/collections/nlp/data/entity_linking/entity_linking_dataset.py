# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import os
import ast
import torch
import random
import array
import pickle as pkl
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from nemo.utils import logging
from nemo.core.classes import Dataset
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import NeuralType, ChannelType, MaskType, LabelsType
from nemo.collections.nlp.parts.utils_funcs import list2str
from nemo.collections.nlp.data.data_utils.data_preprocessing import get_stats, find_newlines, load_data_indices

__all__ = ['EntityLinkingDataset']


class EntityLinkingDataset(Dataset):
    """
    Dataset for second stage pretraining of BERT based encoder 
    models for entity linking. 

    Args:
        tokenizer (obj): huggingface tokenizer,
        data_file (str): path to tab separated column file where data 
            pairs apear in the format 
            concept_ID\tconcept_synonym1\tconcept_synonym2\n
        pair_idx_file (str): path to pickle file containing location
            of data_file newline characters
        max_seq_length (int): maximum length of a concept in tokens
    """

    def __init__(
        self,
        tokenizer: object,
        data_file: str,
        pair_idx_file: Optional[str] = None,
        max_seq_length: Optional[int] = 512,
        ):

        self.tokenizer = tokenizer

        # Try and load pair indices file if already exists
        pair_indices, pair_idx_file = load_data_indices(pair_idx_file, "pair_indices")

        # If pair indices file doesn't exists, generate and store them
        if pair_indices is None:
            logging.info("Getting datafile newline indices")

            with open(data_file, "rb") as f:
                contents = f.read()
                newline_indices = find_newlines(contents)
                newline_indices = array.array("I", newline_indices)

            # Store data file indicies to avoid generating them again
            with open(pair_idx_file, "wb") as f:
                pkl.dump(newline_indices, f)

            pair_indices = newline_indices

        self.pair_indices = pair_indices
        self.data_file = data_file
        self.num_pairs = len(pair_indices)
        self.max_seq_length = max_seq_length
        self.verbose = verbose

        logging.info(f"Loaded dataset with {self.num_pairs} pairs")

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):

        pair_offset = self.pair_indices[idx]

        with open(self.data_file, "rb") as f:
            # Find data pair within datafile using byte offset
            f.seek(pair_offset)
            pair = f.readline()[:-1].decode("utf-8", errors="ignore")
            pair = pair.strip().split("\t")
            cui, sent1, sent2 = pair
    
            # Removing leading C to convert label string to int
            cui = int(cui[1:])

        return (cui, sent1, sent2)


    def _collate_fn(self, batch):
        """collate batch of input_ids, segment_ids, input_mask, and label

        Args:
            batch:  A list of tuples of format (concept_ID, concept_synonym1, concept_synonym2).
        """

        labels, sents1, sents2 = zip(*batch)

        labels = list(labels) 
        labels.extend(labels) # Need to double label list to match each sent

        sents = list(sents1)
        sents.extend(sents2)

        batch = self.tokenizer(sents,
                          add_special_tokens = True,
                          padding = True,
                          truncation = True,
                          max_length = self.max_seq_length,
                          return_token_type_ids = True,
                          return_attention_mask = True,
                          return_length = True)

        return (
            torch.LongTensor(batch["input_ids"]),
            torch.LongTensor(batch["token_type_ids"]),
            torch.LongTensor(batch["attention_mask"]),
            torch.LongTensor(labels),
        )
