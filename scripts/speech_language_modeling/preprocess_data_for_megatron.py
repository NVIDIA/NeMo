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

"""Processing data for megatron pretraining.

It can be used to convert the text data into indexed dataset for GPT models.
For other models (BERT, T5, RETRO), refer to /scripts/nlp_language_modeling/preprocess_data_for_megatron.py

Example script to preprocess the loose JSON file for GPT model

Example json line:
{"audio_codes": "path_to_audio_codes.npz", "text": "corresponding text"}
npz file should contain a numpy array of shape [n_codebooks, n_frames] under the key "codes"

```python
python scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input=PATH_TO_THE_RETRIEVAL_DB_LOOSE_JSON_FILE \
    --json-keys=text \
    --tokenizer-library=megatron \
    --tokenizer-type=GPT2BPETokenizer \
    --dataset-impl=mmap \
    --merge-file=YOUR_MERGE_FILE \
    --vocab-file=YOUR_VOCAB_FILE \
    --output-prefix=YOUR_DATA_PREFIX \
    --append-eod \
    --flatten_audio_codebooks \
    --n_codebooks_to_use=1 \
    --workers=48
```
"""

import argparse
import gzip
import json
import multiprocessing
import os
import pathlib
import sys
import time

import ftfy
import torch
import numpy as np

from nemo.collections.nlp.data.language_modeling.megatron import indexed_dataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

try:
    import nltk

    nltk_available = True
except ImportError:
    nltk_available = False

# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""


class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


def get_tokenizer(args):
    tokenizer = get_nmt_tokenizer(
        library=args.tokenizer_library,
        model_name=args.tokenizer_type,
        tokenizer_model=args.tokenizer_model,
        vocab_file=args.vocab_file,
        merges_file=args.merge_file,
        delimiter=args.delimiter,
    )
    if args.need_pad_id:
        if not hasattr(tokenizer, "pad_id"):
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
        elif hasattr(tokenizer, "pad_id") and (tokenizer.pad_id is None or tokenizer.pad_id < 0):
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
    return tokenizer

# TODO: this Encoder is copy of Encoder in ../nlp_language_modeling/preprocess_data_for_megatron.py
# with some code refactor, should merge them together
class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = get_tokenizer(self.args)

        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            splitter = nltk.load("tokenizers/punkt/english.pickle")
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text=splitter._params, lang_vars=CustomLanguageVars()
                )
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def process(self, text):
        text = text.strip()
        if self.args.apply_ftfy:
            text = ftfy.fix_text(text)
        doc_ids = []
        for sentence in Encoder.splitter.tokenize(text):
            sentence_ids = Encoder.tokenizer.text_to_ids(sentence)
            if len(sentence_ids) > 0:
                doc_ids.append(sentence_ids)
        if len(doc_ids) > 0 and self.args.append_eod:
            doc_ids[-1].append(Encoder.tokenizer.eos_id)
        return doc_ids

    def encode(self, json_line):
        if not self.args.text_file:
            data = json.loads(json_line)
            ids = {}
            for key in self.args.json_keys:
                text = data[key]
                doc_ids = self.process(text)
                """
                if self.args.apply_ftfy:
                    text = ftfy.fix_text(text)
                doc_ids = []
                for sentence in Encoder.splitter.tokenize(text):
                    sentence_ids = Encoder.tokenizer.text_to_ids(sentence)
                    if len(sentence_ids) > 0:
                        doc_ids.append(sentence_ids)
                if len(doc_ids) > 0 and self.args.append_eod:
                    doc_ids[-1].append(Encoder.tokenizer.eos_id)
                """
                ids[key] = doc_ids
        else:
            data = json_line
            ids = {}
            doc_ids = self.process(data)
            """
            text = data.strip()
            if self.args.apply_ftfy:
                text = ftfy.fix_text(text)
            doc_ids = []
            for sentence in Encoder.splitter.tokenize(text):
                sentence_ids = Encoder.tokenizer.text_to_ids(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.append(sentence_ids)
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eos_id)
            """
            ids['text'] = doc_ids
        return ids, len(json_line)


class AudioEncoder(object):
    def __init__(self, args):
        self.args = args
        self.token_id_offset = args.audio_token_id_offset

    def initializer(self):
        pass

    def flatten_codebooks(self, codes):
        """flatten codebooks
        """
        if len(codes.shape) == 1:
            return codes
        
        codes = codes.copy()
        for n in range(1, codes.shape[0]):
            codes[n, :] += self.codebook_size * n
        flat_codes = codes.ravel("F")
        return flat_codes
    
    def process(self, file_path):
        """load pt/npz file from filepath
        """

        # sometime there is error in reading npz files
        try:
            if file_path.endswith(".npz"):
                codes = np.load(file_path)
                codes = codes["codes"]  # [n_codebooks, n_frames]
            elif file_path.endswith(".pt"):
                codes = torch.load(file_path, map_location='cpu')
                codes = codes.to(torch.int32).squeeze(0).numpy()  # [n_codebooks, n_frames]
            else:
                raise ValueError(f"file_path must be either .npz or .pt, got {file_path}")
        except:
            print(f"Error in reading audio codes from {file_path}")
            return []

        # if semantic codes, expand dimension
        # hack to make the assert below work, fix later
        if len(codes.shape) == 1:
            codes = np.expand_dims(codes, axis=0)  
        
        # if n_codebooks_to_use is greater than the number of codebooks in the file, raise error
        if self.args.n_codebooks_to_use is not None and self.args.n_codebooks_to_use > codes.shape[0]:
            raise ValueError(
                f"n_codebooks_to_use ({self.args.n_codebooks_to_use}) is greater than the number of codebooks in the file ({codes.shape[0]})."
            )
        
        assert len(codes.shape) == 2, f"codes must be 2D, got {len(codes.shape)}"

        if self.args.n_codebooks_to_use is not None:
            codes = codes[:self.args.n_codebooks_to_use, :]    # [N, T] if you use [:N, :], you will get [N, T] even if N=1
        codes = np.squeeze(codes)  # remove the empty dimension if num_codebooks == 1

        if self.token_id_offset > 0:
            codes = codes + self.token_id_offset
        
        # flatten
        if self.args.flatten_audio_codebooks and (self.args.n_codebooks_to_use > 1):
            codes = self.flatten_codebooks(codes)      # [T]  
        
        # pack it in right format append eod if needed
        doc_ids = []
        if len(codes) > 0:
            doc_ids.append(codes)       # list of numpy arrays

        if len(doc_ids) > 0 and self.args.append_eod_for_audio:
            # to deal with single codebook vs multiple codebooks
            audio_eod_shape = (doc_ids[-1].shape[0], 1) if len(doc_ids[-1].shape) == 2 else (1,) 
            audio_eod_dtype = doc_ids[-1].dtype
            audio_eod = np.full(audio_eod_shape, Encoder.tokenizer.eos_id, dtype=audio_eod_dtype)
            
            # concat audio_eod (which is same as eos_id from text tokenizer for now)
            doc_ids[-1] = np.concatenate([doc_ids[-1], audio_eod], axis=-1)
            # doc_ids[-1].append(Encoder.tokenizer.eos_id)

        return doc_ids    


    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        for key in self.args.json_keys:
            audio = data[key]       # audio codes filepath
            doc_ids = self.process(audio)     # [T] or [N, T]
            ids[key] = doc_ids
        return ids, len(json_line)      # TODO: len(json_line) is not meaningful for audio data, change it
 

class AudioTextEncoder(object):
    def __init__(self, args):
        self.args = args
        self.text_encoder = Encoder(args)
        self.audio_encoder = AudioEncoder(args)

    def initializer(self):
        self.text_encoder.initializer()
        self.audio_encoder.initializer()

    def encode(self, json_line):
        if self.args.text_file:
            raise ValueError("text_file is not supported for AudioTextEncoder")
        data = json.loads(json_line)
        ids = {}
        for key in self.args.json_keys:
            if key == "text":
                text = data[key]
                doc_ids = self.text_encoder.process(text)
                processed = len(json_line)

            if key == "audio_codes":
                audio = data[key]
                doc_ids = self.audio_encoder.process(audio)
                processed = len(json_line)  # TODO: Not meaningful for audio data

            # paired audio text
            if key == "audio_text":
                # audio
                audio = data["audio_codes"]
                audio_ids = self.audio_encoder.process(audio)   # list of arrays; change name to audio_doc_ids
                # text
                text = data["text"]
                text_ids = self.text_encoder.process(text)      # list of arrays; change name to text_doc_ids
                text_ids = [np.array(t) for t in text_ids]      

                # if audio_ids is multi-dim, pad text_ids
                # just look at the last element of audio_ids to get the shape
                # other elements should have the same shape
                if len(audio_ids[-1].shape) > 1:
                    for i, t in enumerate(text_ids):
                        # expand text_ids dims to match audio_ids
                        if len(t.shape) == 1:
                            text_ids[i] = np.expand_dims(text_ids[i], axis=0)
                        pad_matrix_shape = (audio_ids[-1].shape[0]-1, t.shape[-1])
                        pad_matrix = np.full(pad_matrix_shape, Encoder.tokenizer.pad_id, dtype=t.dtype)
                        
                        # make sure that pad_matrix is not null : corner case
                        if pad_matrix.size > 0:
                            text_ids[i] = np.concatenate([text_ids[i], pad_matrix], axis=0)

                # concat corresponding audio and text
                if not len(audio_ids) == len(text_ids):
                    raise ValueError(f"len(audio_ids) ({len(audio_ids)}) != len(text_ids) ({len(text_ids)}). you may have used split_sentences flag for text")
                
                doc_ids = []
                for i in range(len(audio_ids)):
                    doc_ids.append(np.concatenate([audio_ids[i], text_ids[i]], axis=-1))

                processed = len(json_line)  # TODO: Not meaningful for audio_text data

            ids[key] = doc_ids
        
        return ids, processed


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the input json or json.gz file. If preprocessing an entire folder, set the --preproc-folder flag and provide the path to the folder in this arg.',
    )
    group.add_argument(
        '--json-keys', nargs='+', default=['text'], help='space separate listed of keys to extract from json'
    )
    group.add_argument('--split-sentences', action='store_true', help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true', help='Keep newlines between sentences when splitting.')
    group.add_argument('--text_file', action='store_true', help='Use text file instead of json.')
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument(
        '--tokenizer-library',
        type=str,
        required=True,
        choices=['yttm', 'sentencepiece', 'megatron', 'huggingface', 'tabular'],
        help='What tokenizer library to use.',
    )
    group.add_argument(
        '--tokenizer-type', type=str, default=None, help='What type of tokenizer to use.',
    )
    group.add_argument(
        '--tokenizer-model', type=str, default=None, help='Path to tokenizer model.',
    )
    group.add_argument('--vocab-file', type=str, default=None, help='Path to the vocab file')
    group.add_argument('--files-filter', type=str, default='**/*.json*', help='files filter str')
    group.add_argument('--merge-file', type=str, default=None, help='Path to the BPE merge file (if necessary).')
    group.add_argument('--delimiter', type=str, default=None, help='delimiter used for tabular tokenizer')
    group.add_argument('--append-eod', action='store_true', help='Append an <eod> token to the end of a document.')
    group.add_argument('--retrieval-db', action='store_true', help='Dataset used for retrieval.')
    group.add_argument('--need-pad-id', action='store_true', help='Whether we need the pad id for the tokenizer')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True, help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap', choices=['lazy', 'cached', 'mmap', 'retmmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1, help='Number of worker processes to launch')
    group.add_argument('--chunk_size', type=int, default=64, help='chunk size used for retrieval')
    group.add_argument(
        '--chunk_stride_size', type=int, default=64, help='the stride size for neighbor chunks used for retrieval'
    )

    group.add_argument('--log-interval', type=int, default=100, help='Interval between progress updates')
    group.add_argument(
        '--preproc-folder',
        action='store_true',
        help='If set, will preprocess all .json or .json.gz files into a single .bin and .idx file. Folder path provided via the --input arg',
    )
    group.add_argument('--apply-ftfy', action='store_true', help='If set, will apply ftfy to the input text')
    
    # audio_codes related args
    group = parser.add_argument_group(title='audio Tokenizer')
    group.add_argument('--flatten_audio_codebooks', action='store_true', help='flatten audio codebooks')
    group.add_argument('--n_codebooks_to_use', type=int, default=None, help='number of codebooks to use')
    group.add_argument('--append_eod_for_audio', action='store_true', help='append eod for audio')
    group.add_argument('--audio_token_id_offset', type=int, default=256003, help='audio token id offset')

    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type is not None and args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0
    # TODO: There are dependencies b/w libraries and model files / tokenizer type strings to check.
    assert args.tokenizer_type is not None or args.tokenizer_model is not None
    
    if args.dataset_impl == 'mmap':     # if you are not flattening use 'lazy' or 'cached' which use 'IndexedDatasetBuilder'
        assert args.flatten_audio_codebooks or args.n_codebooks_to_use==1 , "mmap need --flatten_audio_codebooks flag or n_codebooks_to_use=1"

    
    return args


def main():
    args = get_args()
    startup_start = time.time()
    if args.preproc_folder:
        print('Searching folder for .json or .json.gz files...')
        assert os.path.exists(args.input), f'Folder does not exist: {args.input}'
        json_files = (str(f) for f in pathlib.Path(args.input).glob(args.files_filter))
        json_files = [f for f in json_files if f.endswith('.json') or f.endswith('.json.gz')]
        if len(json_files) == 0:
            raise FileNotFoundError('No .json or .json.gz files found in folder.')
        else:
            print(f'Found {len(json_files)} .json or .json.gz files.')
    else:
        assert os.path.exists(args.input), f'File does not exist: {args.input}'
        json_files = [args.input]

    # encoder = AudioEncoder(args)    
    encoder = AudioTextEncoder(args)

    if args.dataset_impl == 'retmmap':
        assert args.need_pad_id, "retmmap need --need_pad_id flag"
    tokenizer = get_tokenizer(args)

    level = "document"
    if args.split_sentences:
        level = "sentence"

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix, key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix, key, level)
        builders[key] = indexed_dataset.make_builder(
            output_bin_files[key],
            impl=args.dataset_impl,
            chunk_size=args.chunk_size,
            pad_id=tokenizer.pad_id if hasattr(tokenizer, "pad_id") else 0,
            retrieval_db=args.retrieval_db,
            vocab_size=tokenizer.vocab_size,
            stride=args.chunk_stride_size,
        )

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)

    for idx, json_file in enumerate(json_files):
        print(f'Processing file {json_file} {idx + 1}/{len(json_files)}')
        if json_file.endswith('.gz'):
            fin = gzip.open(json_file, 'r')
        else:
            fin = open(args.input, 'r', encoding='utf-8')

        encoded_docs = pool.imap(encoder.encode, fin, 25)

        for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            for key, sentences in doc.items():
                if len(sentences) == 0:
                    continue
                for sentence in sentences:
                    builders[key].add_item(torch.IntTensor(sentence))
                builders[key].end_document()
            if i % args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {i} documents", f"({i/elapsed} docs/s, {mbs} MB/s).", file=sys.stderr)

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])


if __name__ == '__main__':
    main()
