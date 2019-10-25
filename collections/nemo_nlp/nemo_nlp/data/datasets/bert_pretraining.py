# =============================================================================
# Copyright 2019 AI Applications Design Team at NVIDIA. All Rights Reserved.
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
# =============================================================================
"""Pytorch Dataset for training BERT."""

import array
import glob
import os
import pickle
import random

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class BertPretrainingDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 dataset,
                 max_seq_length=128,
                 mask_probability=0.15,
                 sentence_idx_file=None):
        self.tokenizer = tokenizer
        self.token_cls = tokenizer.token_to_id("[CLS]")
        self.token_sep = tokenizer.token_to_id("[SEP]")

        # Loading enormous datasets into RAM isn't always feasible -- for
        # example, the pubmed corpus is 200+ GB, which doesn't fit into RAM on
        # most computers. To get around this, we store the indices of newlines
        # in each file so we can seek to and retrieve sentences immediately
        # from main memory when needed during training.

        if sentence_idx_file is None:
            data_dir = dataset[:dataset.rfind('/')]
            mode = dataset[dataset.rfind('/')+1:dataset.rfind('.')]
            sentence_idx_file = f"{data_dir}/{mode}_sentence_indices.pkl"

        if os.path.isfile(sentence_idx_file):
            # If the sentence indices file already exists, load from it
            with open(sentence_idx_file, "rb") as f:
                sentence_indices = pickle.load(f)
        else:
            # Otherwise, generate and store sentence indices
            sentence_indices = {}
            total_tokens = 0
            used_tokens = 0

            # Finds all of the newline indices in a string
            def find_newlines(contents):
                nonlocal used_tokens, total_tokens
                start = 0

                while True:
                    try:
                        # index and split are much faster than Python for loops
                        new_start = contents.index(b"\n", start)
                        line = contents[start:new_start] \
                            .replace(b"\xc2\x99", b" ") \
                            .replace(b"\xc2\xa0", b" ")
                        num_tokens = len(line.split())

                        # Ensure the line has at least max_seq_length tokens
                        if num_tokens >= max_seq_length:
                            yield start - 1
                            used_tokens += num_tokens

                        total_tokens += num_tokens
                        start = new_start + 1
                    except ValueError:
                        break

            if os.path.isdir(dataset):
                dataset_pattern = os.path.join(dataset, "**", "*.txt")
                filenames = glob.glob(dataset_pattern, recursive=True)
            else:
                filenames = [dataset]

            for filename in tqdm(filenames):
                with open(filename, "rb") as f:
                    contents = f.read()
                    newline_indices = find_newlines(contents)

                if os.path.isdir(dataset):
                    # Only keep the parts of the filepath that are invariant to
                    # the dataset's location on disk
                    filename = os.path.basename(filename)

                # In python, arrays are much more space-efficient than lists
                sentence_indices[filename] = array.array("I", newline_indices)

            # Save sentence indices so we don't have to do this again
            with open(sentence_idx_file, "wb") as f:
                pickle.dump(sentence_indices, f)

            print(f"Used {used_tokens} of total {total_tokens} tokens")

        corpus_size = 0
        empty_files = []

        # Find total number of newlines across entire corpus and remove files
        # without any newlines
        for filename in sentence_indices:
            if len(sentence_indices[filename]) <= 1:
                empty_files.append(filename)
            else:
                corpus_size += len(sentence_indices[filename])

        for filename in empty_files:
            del sentence_indices[filename]

        self.corpus_size = corpus_size
        self.dataset = dataset
        self.filenames = list(sentence_indices.keys())
        self.mask_probability = mask_probability
        self.max_seq_length = max_seq_length
        self.sentence_indices = sentence_indices
        self.vocab_size = self.tokenizer.vocab_size

    def __len__(self):
        return self.corpus_size

    def __getitem__(self, idx, min_doc_length=16):
        # Each sequence has three special tokens, as follows:
        # [CLS] <document a> [SEP] <document b> [SEP]
        num_special_tokens = 3

        # TODO: Make seq_length = 512 for the last 10% of epochs, as specified
        # in BERT paper
        seq_length = self.max_seq_length - num_special_tokens

        a_length = random.randrange(min_doc_length,
                                    seq_length - min_doc_length + 1)
        b_length = seq_length - a_length

        a_filename = random.choice(self.filenames)
        a_line = random.choice(self.sentence_indices[a_filename])

        def get_document(filepath, line):
            # Retrieve a specific line from a file and return as a document
            if os.path.isdir(self.dataset):
                filepath = os.path.join(self.dataset, filepath)

            with open(filepath, "rb") as f:
                # Add one to go to the character after the newline
                f.seek(line + 1)

                # Read line, remove newline, and decode as UTF8
                doc_text = f.readline()[:-1].decode("utf-8", errors="ignore")

            document = self.tokenizer.text_to_ids(doc_text)
            assert len(document) >= self.max_seq_length
            return document

        a_document = get_document(a_filename, a_line)

        if random.random() < 0.5:
            # 50% of the time, B is the sentence that follows A
            label = 1

            a_start_idx = random.randrange(len(a_document) - seq_length)
            b_start_idx = a_start_idx + a_length

            b_filename = a_filename
            b_line = a_line
            b_document = a_document
        else:
            # The rest of the time, B is a random sentence from the corpus
            label = 0

            a_start_idx = random.randrange(len(a_document) - a_length)

            b_filename = random.choice(self.filenames)
            b_line = random.choice(self.sentence_indices[b_filename])
            b_document = get_document(b_filename, b_line)
            b_start_idx = random.randrange(len(b_document) - b_length)

        # Process retrieved documents for use in training
        a_ids = a_document[a_start_idx:a_start_idx + a_length]
        b_ids = b_document[b_start_idx:b_start_idx + b_length]
        output_ids = [self.token_cls] + a_ids + [self.token_sep] \
            + b_ids + [self.token_sep]

        input_ids, output_mask = self.mask_ids(output_ids)

        output_mask = np.array(output_mask, dtype=np.float32)
        input_mask = np.ones(self.max_seq_length, dtype=np.float32)

        input_type_ids = np.zeros(self.max_seq_length, dtype=np.int)
        input_type_ids[a_length + 2:seq_length + 3] = 1

        return (np.array(input_ids),
                input_type_ids,
                input_mask,
                np.array(output_ids),
                output_mask,
                label)

    def mask_ids(self, ids):
        """
        Args:
          tokens: list of tokens representing a chunk of text

        Returns:
          masked_tokens: list of input tokens with some of the entries masked
            according to the following protocol from the original BERT paper:
            each token is masked with a probability of 15% and is replaced with
            1) the [MASK] token 80% of the time,
            2) random token 10% of the time,
            3) the same token 10% of the time.
          output_mask: list of binary variables which indicate what tokens has
            been masked (to calculate the loss function for these tokens only)
        """
        masked_ids = []
        output_mask = []
        for i in ids:
            if random.random() < self.mask_probability:
                output_mask.append(1)
                if random.random() < 0.8:
                    masked_ids.append(self.tokenizer.special_tokens["[MASK]"])
                elif random.random() < 0.5:
                    masked_ids.append(random.randrange(self.vocab_size))
                else:
                    masked_ids.append(i)
            else:
                masked_ids.append(i)
                output_mask.append(0)
        return masked_ids, output_mask
