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
from nemo.collections.nlp.parts.utils_funcs import list2str, find_newlines, load_data_indices
from nemo.collections.nlp.data.data_utils.data_preprocessing import get_stats


class SelfAlignmentPretrainingDataset(Dataset):
    """
    Dataset for second stage pretraining of BERT based models for entity linking 
    """

    def __init__(
        self,
        tokenizer: object,
        data_file: str,
        pair_idx_file: Optional[str] = None,
        max_seq_length: Optional[int] = 512,
        verbose: Optional[bool] = False,
        ):

        self.tokenizer = tokenizer

        # Try and load pair indices file if already exists
        pair_indices, pair_idx_file = load_data_indices(pair_idx_file, "pair_indices")

        # If pair indices file doesn't exists, generate and store them
        if pair_indices is None:
            with open(data_file, "rb") as f:
                contents = f.read()
                newline_indices = find_newlines(contents)
                newline_indices = array.array("I", newline_indices)

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
            batch:  A list of tuples of (input_ids, segment_ids, input_mask, label).
        """

        labels, sents1, sents2 = zip(*batch)

        labels = list(labels) 
        labels.extend(labels) # Need to double label list to match each sent

        sents = list(sents1)
        sents.extend(sents2)

        batch = self.preprocess_batch(sents)

        return (
            torch.LongTensor(batch["input_ids"]),
            torch.LongTensor(batch["token_type_ids"]),
            torch.LongTensor(batch["attention_mask"]),
            torch.LongTensor(labels),
        )


    def preprocess_batch(self, sents):
        """Encode a list of sentences into a list of tuples of (input_ids, segment_ids, input_mask).
           """
        batch = self.tokenizer(sents,
                          add_special_tokens = True,
                          padding = True,
                          truncation = True,
                          max_length = self.max_seq_length,
                          return_token_type_ids = True,
                          return_attention_mask = True,
                          return_length = True)

        if self.verbose:
            logging.info("***Tokenzer Example***")
            logging.info(f"example sentence: {sents[0]}")
            logging.info("subtokens: %s" % " ".join(self.tokenizer.tokenize(sents[0])))
            logging.info("input_ids: %s" % list2str(batch["input_ids"][0]))
            logging.info("segment_ids: %s" % list2str(batch["token_type_ids"][0]))
            logging.info("input_mask: %s" % list2str(batch["attention_mask"][0]))
    
        return batch


    #TODO: Implement this correctly
    #@typecheck()
    #def output_types(self) -> Optional[Dict[str, NeuralType]]:
    #    """Returns definitions of module output ports.
    #           """
    #    return {
    #        'input_ids': NeuralType(('B', 'T'), ChannelType()),
    #        'segment_ids': NeuralType(('B', 'T'), ChannelType()),
    #        'input_mask': NeuralType(('B', 'T'), MaskType()),
    #        'label': NeuralType(('B',), LabelsType()),
    #    }
