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
                 name,
                 sentence_indices_filename=None,
                 max_length=128,
                 mask_probability=0.15,
                 short_seq_prob=0.1):
        self.tokenizer = tokenizer

        if sentence_indices_filename is None:
            sentence_indices_filename = "{}_sentence_indices.pkl".format(name)

        # Loading enormous datasets into RAM isn't always feasible -- for
        # example, the pubmed corpus is 200+ GB, which doesn't fit into RAM on
        # most computers. To get around this, we store the indices of newlines
        # in each file so we can seek to and retrieve sentences immediately
        # from main memory when needed during training.

        if os.path.isfile(sentence_indices_filename):
            # If the sentence indices file already exists, load from it
            with open(sentence_indices_filename, "rb") as f:
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
                            .replace(b"\xc2\xa0", b" ") \
                            .replace(b"\xa0", b" ")
                        num_tokens = len(line.split())

                        yield new_start

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
            with open(sentence_indices_filename, "wb") as f:
                pickle.dump(sentence_indices, f)

            print("Used {} tokens of total {}".format(used_tokens,
                                                      total_tokens))

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
        self.max_length = max_length
        self.sentence_indices = sentence_indices
        self.vocab_size = self.tokenizer.vocab_size
        self.short_seq_prob = short_seq_prob
        

    def __len__(self):
        return self.corpus_size

    def __getitem__(self, idx):
        # Each sequence has three special tokens, as follows:
        # [CLS] <document a> [SEP] <document b> [SEP]
        num_special_tokens = 3

        max_num_tokens = self.max_length - num_special_tokens
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob:
          target_seq_length = random.randint(2, max_num_tokens)

        a_filename = random.choice(self.filenames)
        a_line_idx = random.choice(range(len(self.sentence_indices[a_filename])))

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

          return document


        def match_target_seq_length(document, target_seq_length, filename, 
                                    line_idx, sentence_indices):
          # match the target seq_length
          for _ in range(10):
            # if line_idx + 1 is larger than the file, then we don't match seq_length
            # - it'll be a shorter seq_length.
            if len(document) < target_seq_length and \
              line_idx + 1 < len(sentence_indices[filename]):
              if sentence_indices[filename][line_idx + 1] < \
                                          self.sentence_indices[filename][-1]:
                line_idx += 1
                document += get_document(filename, 
                                  self.sentence_indices[filename][line_idx])
              else:
                line_idx = random.choice(range(len(sentence_indices[filename])))
                document = get_document(filename, 
                                        sentence_indices[filename][line_idx])
                document, line_idx = match_target_seq_length(document, 
                          target_seq_length, filename, line_idx, sentence_indices)
            else:
              break

          return document, line_idx
         

        target_seq_length_a = int(round(target_seq_length * 0.6))
        target_seq_length_b = target_seq_length - target_seq_length_a
        a_document = get_document(a_filename, 
                                  self.sentence_indices[a_filename][a_line_idx])
        a_document, a_line_idx = match_target_seq_length(a_document, 
                                              target_seq_length_a, a_filename,
                                              a_line_idx, self.sentence_indices)


        if self.sentence_indices[a_filename][a_line_idx] >= \
                    self.sentence_indices[a_filename][-1] or \
                    random.random() < 0.5:

          # About 50% of the time, B is a random sentence from the corpus
          label = 0

          for _ in range(10):
            b_filename = random.choice(self.filenames)
            b_line_idx = random.choice(range(len(self.sentence_indices[b_filename])))
            if b_filename != a_filename:
              break
            else:
              new_idx = self.sentence_indices[b_filename][b_line_idx]
              old_idx = self.sentence_indices[b_filename][a_line_idx]
              if (new_idx < old_idx - self.max_length) or \
                (new_idx > old_idx + self.max_length):
                break
              else:
                pass

          b_document = get_document(b_filename, 
                                    self.sentence_indices[b_filename][b_line_idx])
          b_document, b_line_idx = match_target_seq_length(b_document, 
                                              target_seq_length_b, b_filename, 
                                              b_line_idx, self.sentence_indices)

        else:
          label = 1

          b_filename = a_filename
          b_line_idx = a_line_idx + 1
          b_document = get_document(a_filename, 
                                    self.sentence_indices[a_filename][b_line_idx])
          b_document, b_line_idx = match_target_seq_length(b_document, 
                                              target_seq_length_b, b_filename,
                                              b_line_idx, self.sentence_indices)


        def truncate_seq_pair(a_document, b_document, max_num_tokens):
          # Truncates a pair of sequences to a maximum sequence length
          while True:
            total_length = len(a_document) + len(b_document)
            if total_length <= max_num_tokens:
              break

            trunc_document = a_document if len(a_document) > len(b_document) \
                                        else b_document
            assert len(trunc_document) >= 1

            if random.random() < 0.5:
              del trunc_document[0]
            else:
              trunc_document.pop()


        truncate_seq_pair(a_document, b_document, max_num_tokens)


        output_ids = [self.tokenizer.special_tokens["[CLS]"]] + a_document + \
                     [self.tokenizer.special_tokens["[SEP]"]] + b_document + \
                     [self.tokenizer.special_tokens["[SEP]"]]

        input_ids, output_mask = self.mask_ids(output_ids)

        output_mask = np.array(output_mask, dtype=np.float32)
        input_mask = np.zeros(self.max_length, dtype=np.float32)
        input_mask[:len(input_ids)] = 1

        input_type_ids = np.zeros(self.max_length, dtype=np.int)
        input_type_ids[len(a_document) + 2:len(output_ids) + 1] = 1

        while len(input_ids) < self.max_length:
          input_ids.append(0)
          output_ids.append(0)
          output_mask = np.append(output_mask, [0])

        return np.array(input_ids), input_type_ids,\
            np.array(input_mask, dtype=np.float32),\
            np.array(output_ids), np.array(output_mask, dtype=np.float32), label


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

      cand_indexes = []
      for (i, id) in enumerate(ids):

        if len(cand_indexes) >= 1 and \
          not self.tokenizer.ids_to_tokens([id])[0].startswith('\u2581'):
          cand_indexes[-1].append(id)
        else:
          cand_indexes.append([id])
      
      masked_ids = []
      output_mask = []
      for cand_index in cand_indexes:
        if (random.random() < self.mask_probability) and \
          cand_index[0] != self.tokenizer.special_tokens["[CLS]"] and \
          cand_index[0] != self.tokenizer.special_tokens["[SEP]"]:
          if random.random() < 0.8:
            for cand_index_i in cand_index:
              output_mask.append(1)
              masked_ids.append(self.tokenizer.special_tokens["[MASK]"])
          elif random.random() < 0.5:
            for cand_index_i in cand_index:
              output_mask.append(1)
              for _ in range(10):
                random_word = random.randrange(self.vocab_size)
                if random_word != self.tokenizer.special_tokens["[SEP]"] and \
                  random_word != self.tokenizer.special_tokens["[CLS]"]:
                  break
              masked_ids.append(random_word)
          else:
            for cand_index_i in cand_index:
              output_mask.append(1)
              masked_ids.append(cand_index_i)
        else:
          for cand_index_i in cand_index:  
            masked_ids.append(cand_index_i)
            output_mask.append(0) 
      
      return masked_ids, output_mask
