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
# ==============================================================================
"""Pytorch Dataset for training Neural Machine Translation."""
import collections.nlp.data.datasets.datasets_utils
import glob
import os
import pickle
import re
from collections.nlp.data.datasets.datasets_utils import DATABASE_EXISTS_TMP, download_wkt2
from collections.nlp.utils.common_nlp_utils import if_exist

import numpy as np
from sentencepiece import SentencePieceTrainer as SPT
from torch.utils.data import Dataset
from tqdm import tqdm

import nemo

__all__ = ['LanguageModelingDataset']


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
        src_mask = (src_ids != self.tokenizer.pad_id()).astype(np.float32)
        return src_ids, src_mask, labels


class LanguageModelDataDesc:
    def __init__(self, dataset_name, data_dir, do_lower_case):
        if dataset_name == 'wikitext-2':
            if not os.path.exists(data_dir):
                data_dir = download_wkt2(data_dir)
            self.vocab_size = create_vocab_lm(data_dir, do_lower_case)
            self.data_dir = data_dir
        else:
            nemo.logging.warning(
                "Looks like you passed a dataset name that isn't "
                "already supported by NeMo. Please make sure that "
                "you build the preprocessing method for it."
            )


def create_vocab_mlm(
    data_dir, vocab_size, sample_size, special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'], train_file=''
):
    vocab = special_tokens[:]
    bert_dir = f'{data_dir}/bert'
    if if_exist(bert_dir, ['tokenizer.model']):
        nemo.logging.info(DATABASE_EXISTS_TMP.format('WikiText_BERT', bert_dir))
        return data_dir, f'{bert_dir}/tokenizer.model'
    nemo.logging.info(f'Processing WikiText dataset and store at {bert_dir}')
    os.makedirs(bert_dir, exist_ok=True)

    if not train_file:
        files = glob.glob(f'{data_dir}/*.txt')
        train_file = f'{bert_dir}/merged.txt'
        nemo.logging.info(f"Merging {len(files)} txt files into {train_file}")

        with open(train_file, "w") as merged:
            for file in tqdm(files):
                with open(file, 'r') as inf:
                    content = inf.read().strip()
                merged.write(content + '\n\n\n')
    else:
        train_file = f'{data_dir}/{train_file}'

    cmd = (
        f"--input={train_file} --model_prefix={bert_dir}/tokenizer "
        f"--vocab_size={vocab_size - len(vocab)} "
        f"--input_sentence_size={sample_size} "
        f"--shuffle_input_sentence=true --hard_vocab_limit=false "
        f"--bos_id=-1 --eos_id=-1"
    )
    SPT.Train(cmd)

    # Add BERT control symbols
    tokens = []

    with open(f"{bert_dir}/tokenizer.vocab", "r") as f:
        f.readline()  # skip first <unk> token

        # Read tokens from each line and parse for vocab
        for line in f:
            piece = line.split("\t")[0]
            token = piece[1:] if piece.startswith("▁") else f"##{piece}"
            tokens.append(token)

    vocab.extend(tokens)

    # Save vocabulary to output file
    with open(f'{bert_dir}/vocab.txt', "w") as f:
        for token in vocab:
            f.write(f"{token}\n".format())
    return data_dir, f'{bert_dir}/tokenizer.model'


def dataset_to_ids(dataset, tokenizer, cache_ids=False, add_bos_eos=True):
    """
    Reads dataset from file line by line, tokenizes each line with tokenizer,
    and returns list of lists which corresponds to ids of tokenized strings.

    Args:
        dataset: path to dataset
        tokenizer: tokenizer to convert text into ids
        cache_ids: if True, ids are saved to disk as pickle file
            with similar name (e.g., data.txt --> data.txt.pkl)
        add_bos_eos: bool, whether to add <s> and </s> symbols (e.g., for NMT)
    Returns:
        ids: list of ids which correspond to tokenized strings of the dataset
    """

    cached_ids_dataset = dataset + str(".pkl")
    if os.path.isfile(cached_ids_dataset):
        nemo.logging.info("Loading cached tokenized dataset ...")
        ids = pickle.load(open(cached_ids_dataset, "rb"))
    else:
        nemo.logging.info("Tokenizing dataset ...")
        data = open(dataset, "rb").readlines()
        ids = []
        for sentence in data:
            sent_ids = tokenizer.text_to_ids(sentence.decode("utf-8"))
            if add_bos_eos:
                sent_ids = [tokenizer.bos_id()] + sent_ids + [tokenizer.eos_id()]
            ids.append(sent_ids)
        if cache_ids:
            nemo.logging.info("Caching tokenized dataset ...")
            pickle.dump(ids, open(cached_ids_dataset, "wb"))
    return ids


def create_vocab_lm(data_dir, do_lower_case):
    if if_exist(data_dir, ['train.txt', 'vocab.txt']):
        nemo.logging.info("Vocabulary has been created.")
        with open(os.path.join(data_dir, 'vocab.txt'), 'r') as f:
            vocab_size = len(f.readlines())
        return vocab_size

    nemo.logging.info(f'Creating vocabulary from training data at {data_dir}')

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
    nemo.logging.info(f"Created vocabulary of size {len(vocab)}")

    return len(vocab)
