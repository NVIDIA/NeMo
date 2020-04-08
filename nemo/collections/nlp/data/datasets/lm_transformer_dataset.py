# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
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

"""Pytorch Dataset for training Neural Machine Translation."""
import os
import re

import numpy as np
from torch.utils.data import Dataset

from nemo import logging
from nemo.collections.nlp.data.datasets.datasets_utils import dataset_to_ids, if_exist

__all__ = ['LanguageModelingDataset', 'LanguageModelDataDesc']


class LanguageModelingDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_seq_length=512, batch_step=None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.batch_step = batch_step or self.max_seq_length
        ids = dataset_to_ids(dataset, tokenizer, add_bos_eos=False)
        self.ids = np.array([j for i in ids for j in i])

    def __len__(self):
        return (len(self.ids) - self.max_seq_length) // self.batch_step

    def __getitem__(self, idx):
        left = idx * self.batch_step
        right = left + self.max_seq_length
        src_ids = self.ids[left:right]
        labels = self.ids[left + 1 : right + 1]
        src_mask = (src_ids != self.tokenizer.pad_id).astype(np.float32)
        return src_ids, src_mask, labels


class LanguageModelDataDesc:
    def __init__(self, dataset_name, data_dir, do_lower_case):
        if dataset_name == 'wikitext-2':
            if not os.path.exists(data_dir):
                raise FileNotFoundError(
                    "Dataset not found. Run 'get_wkt2.sh DATA_DIR' from examples/nlp/language_modeling"
                )
            self.vocab_size = self.create_vocab_lm(data_dir, do_lower_case)
            self.data_dir = data_dir
        else:
            logging.warning(
                "Looks like you passed a dataset name that isn't "
                "already supported by NeMo. Please make sure that "
                "you build the preprocessing method for it."
            )

    def create_vocab_lm(self, data_dir, do_lower_case):
        if if_exist(data_dir, ['train.txt', 'vocab.txt']):
            logging.info("Vocabulary has been created.")
            with open(os.path.join(data_dir, 'vocab.txt'), 'r') as f:
                vocab_size = len(f.readlines())
            return vocab_size

        logging.info(f'Creating vocabulary from training data at {data_dir}')

        with open(f'{data_dir}/train.txt', 'r') as f:
            txt = f.read()
        if do_lower_case:
            txt = txt.lower()
        lines = re.split(r'[\n]', txt)
        sentences = [line.strip().split() for line in lines if line.strip()]

        vocab = {"[PAD]": 0, "[SEP]": 1, "[CLS]": 2, "[MASK]": 3}
        idx = 4
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1

        with open(f'{data_dir}/vocab.txt', 'w') as f:
            for word in sorted(vocab.keys()):
                f.write(word + '\n')
        logging.info(f"Created vocabulary of size {len(vocab)}")

        return len(vocab)
