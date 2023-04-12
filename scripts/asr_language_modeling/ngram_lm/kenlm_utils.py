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
#

"""
Utility methods to be used for training N-gram LM with KenLM in train_kenlm.py
"""

CHUNK_SIZE = 8192
CHUNK_BUFFER_SIZE = 512

import gzip
import json
import os

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.submodules.ctc_beam_decoding import DEFAULT_TOKEN_OFFSET
from nemo.collections.asr.parts.utils.transcribe_utils import PunctuationCapitalization
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
from nemo.utils import logging

# List of the supported models to be used with N-gram LM and beam search decoding
SUPPORTED_MODELS = {
    'EncDecCTCModelBPE': 'subword',
    'EncDecCTCModel': 'char',
    'EncDecRNNTBPEModel': 'subword',
    'EncDecRNNTModel': 'char',
    'EncDecHybridRNNTCTCBPEModel': 'subword',
}


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1).reshape([x.shape[0], 1])


def get_train_list(args_train_path):

    train_path = []
    for train_item in args_train_path:
        if os.path.isdir(train_item):
            file_list = os.listdir(train_item)
            train_path.extend([os.path.join(train_item, file) for file in file_list])

        elif os.path.isfile(train_item):
            train_path.append(train_item)
    return sorted(train_path)


def setup_tokenizer(tokenizer_model_file):
    """ TOKENIZER SETUP """
    logging.info(f"Loading nemo model '{tokenizer_model_file}' ...")
    if tokenizer_model_file.endswith('.model'):
        tokenizer_nemo = SentencePieceTokenizer(tokenizer_model_file)
    elif tokenizer_model_file.endswith('.nemo'):
        model = nemo_asr.models.ASRModel.restore_from(tokenizer_model_file, map_location=torch.device('cpu'))
    else:
        logging.warning(
            "tokenizer_model_file does not end with .model or .nemo, therefore trying to load a pretrained model with this name."
        )
        model = nemo_asr.models.ASRModel.from_pretrained(tokenizer_model_file, map_location=torch.device('cpu'))

    if tokenizer_model_file.endswith('.model'):
        encoding_level = 'subword'
        is_aggregate_tokenizer = False
    else:
        if type(model.tokenizer).__name__ == 'AggregateTokenizer':
            is_aggregate_tokenizer = True
        else:
            is_aggregate_tokenizer = False

        encoding_level = SUPPORTED_MODELS.get(type(model).__name__, None)
        if not encoding_level:
            logging.warning(
                f"Model type '{type(model).__name__}' may not be supported. Would try to train a char-level LM."
            )
            encoding_level = 'char'

        tokenizer_nemo = model.tokenizer
        del model

    return tokenizer_nemo, encoding_level, is_aggregate_tokenizer


def iter_files(
    dest_path,
    train_path,
    tokenizer,
    encoding_level,
    is_aggregate_tokenizer,
    do_lowercase,
    rm_punctuation,
    separate_punctuation,
    verbose,
):
    if isinstance(dest_path, list):
        train_path = zip(dest_path, train_path)
    else:  # dest_path is stdin of KenLM
        train_path = [(dest_path, path) for path in train_path]

    for dest_path, input_path in train_path:
        dataset = read_train_file(
            input_path,
            do_lowercase,
            rm_punctuation,
            separate_punctuation,
            is_aggregate_tokenizer=is_aggregate_tokenizer,
            verbose=verbose,
        )
        if encoding_level == "subword":
            tokenize_text(
                data=dataset,
                tokenizer=tokenizer,
                path=dest_path,
                chunk_size=CHUNK_SIZE,
                buffer_size=CHUNK_BUFFER_SIZE,
            )
        else:  # encoding_level == "char"
            if isinstance(dest_path, str):
                with open(dest_path, 'w', encoding='utf-8') as f:
                    for line in dataset:
                        f.write(f"{line}\n")
            else:  # write to stdin of KenLM
                for line in dataset:
                    dest_path.write((line + '\n').encode())


def read_train_file(
    path,
    do_lowercase: bool = False,
    rm_punctuation: bool = False,
    separate_punctuation: bool = False,
    is_aggregate_tokenizer: bool = False,
    verbose: int = 0,
):
    lines_read = 0
    text_dataset, lang_dataset = [], []
    punctuation_capitalization = PunctuationCapitalization('.,?')
    if path[-8:] == '.json.gz':
        fin = gzip.open(path, 'r')
    else:
        fin = open(path, 'r', encoding='utf-8')

    if verbose > 0:
        reader = tqdm(iter(lambda: fin.readline(), ''), desc="Read 0 lines", unit=' lines')
    else:
        reader = fin

    for line in reader:
        lang = None
        if line:
            if path[-8:] == '.json.gz':
                line = json.loads(line.decode('utf-8'))['text']
            elif path.endswith('.json'):
                jline = json.loads(line)
                line = jline['text']
                if is_aggregate_tokenizer:
                    lang = jline['lang']

            line_list = line.split("\n")
            if rm_punctuation:
                line_list = punctuation_capitalization.rm_punctuation(line_list)
            if separate_punctuation:
                line_list = punctuation_capitalization.separate_punctuation(line_list)
            if do_lowercase:
                line_list = punctuation_capitalization.do_lowercase(line_list)

            line = " ".join(line_list)
            if line:
                text_dataset.append(line)
                if lang:
                    lang_dataset.append(lang)
                lines_read += 1
                if verbose > 0 and lines_read % 100000 == 0:
                    reader.set_description(f"Read {lines_read} lines")
        else:
            break
    if is_aggregate_tokenizer:
        assert len(text_dataset) == len(
            lang_dataset
        ), f"text_dataset length {len(text_dataset)} and lang_dataset length {len(lang_dataset)} must be the same!"
        return list(zip(text_dataset, lang_dataset))
    else:
        return [[text] for text in text_dataset]


def tokenize_str(texts, tokenizer):
    tokenized_text = []
    for text in texts:
        tok_text = tokenizer.text_to_ids(*text)
        tok_text = [chr(token + DEFAULT_TOKEN_OFFSET) for token in tok_text]
        tokenized_text.append(tok_text)
    return tokenized_text


def tokenize_text(data, tokenizer, path, chunk_size=8192, buffer_size=32):
    dataset_len = len(data)
    current_step = 0
    if isinstance(path, str) and os.path.exists(path):
        os.remove(path)

    with Parallel(n_jobs=-2, verbose=0) as parallel:
        while True:
            start = current_step * chunk_size
            end = min((current_step + buffer_size) * chunk_size, dataset_len)

            tokenized_data = parallel(
                delayed(tokenize_str)(data[start : start + chunk_size], tokenizer)
                for start in range(start, end, chunk_size)
            )

            # Write dataset
            write_dataset(tokenized_data, path)
            current_step += len(tokenized_data)
            logging.info(
                f"Finished writing {len(tokenized_data)} chunks to {path}. Current chunk index = {current_step}"
            )
            del tokenized_data
            if end >= dataset_len:
                break


def write_dataset(chunks, path):
    if isinstance(path, str):
        with open(path, 'a+', encoding='utf-8') as f:
            for chunk_idx in tqdm(range(len(chunks)), desc='Chunk ', total=len(chunks), unit=' chunks'):
                for text in chunks[chunk_idx]:
                    line = ' '.join(text)
                    f.write(f"{line}\n")
    else:  # write to stdin of KenLM
        for chunk_idx in range(len(chunks)):
            for text in chunks[chunk_idx]:
                line = ' '.join(text)
                path.write((line + '\n').encode())
