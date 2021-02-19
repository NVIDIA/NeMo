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

import argparse
import os
import pickle
import time
from pathlib import Path

import youtokentome as yttm

from nemo.collections.nlp.data import TranslationDataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NMT dataset pre-processing')
    parser.add_argument('--shared_tokenizer', action="store_true", help='Whether to share encoder/decoder tokenizers')
    parser.add_argument('--clean', action="store_true", help='Whether to clean dataset based on length diff')
    parser.add_argument('--bpe_dropout', type=float, default=0.1, help='Whether to share encoder/decoder tokenizers')
    parser.add_argument('--src_fname', type=str, required=True, help='Path to the source file')
    parser.add_argument('--tgt_fname', type=str, required=True, help='Path to the target file')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to store dataloader and tokenizer models')
    parser.add_argument('--vocab_size', type=int, default=32000, help='Vocab size after BPE')
    parser.add_argument('--max_seq_length', type=int, default=512, help='Max Sequence Length')
    parser.add_argument('--min_seq_length', type=int, default=1, help='Min Sequence Length')
    parser.add_argument(
        '--tokens_in_batch', type=str, default="8000,12000,16000,40000", help='# Tokens per batch per GPU'
    )

    args = parser.parse_args()
    if args.shared_tokenizer:
        os.system('cat %s %s > %s' % (args.src_fname, args.tgt_fname, '/tmp/concat_dataset.txt'))
        yttm.BPE.train(
            data='/tmp/concat_dataset.txt',
            vocab_size=args.vocab_size,
            model=os.path.join(args.out_dir, 'tokenizer.%d.BPE.model' % (args.vocab_size)),
        )
        encoder_tokenizer_model = os.path.join(args.out_dir, 'tokenizer.%d.BPE.model' % (args.vocab_size))
        decoder_tokenizer_model = os.path.join(args.out_dir, 'tokenizer.%d.BPE.model' % (args.vocab_size))
    else:
        yttm.BPE.train(
            data=args.src_fname,
            vocab_size=args.vocab_size,
            model=os.path.join(args.out_dir, 'tokenizer.encoder.%d.BPE.model' % (args.vocab_size)),
        )

        yttm.BPE.train(
            data=args.tgt_fname,
            vocab_size=args.vocab_size,
            model=os.path.join(args.out_dir, 'tokenizer.decoder.%d.BPE.model' % (args.vocab_size)),
        )
        encoder_tokenizer_model = os.path.join(args.out_dir, 'tokenizer.encoder.%d.BPE.model' % (args.vocab_size))
        decoder_tokenizer_model = os.path.join(args.out_dir, 'tokenizer.decoder.%d.BPE.model' % (args.vocab_size))

    encoder_tokenizer = get_tokenizer(
        tokenizer_name='yttm', tokenizer_model=encoder_tokenizer_model, bpe_dropout=args.bpe_dropout
    )

    decoder_tokenizer = get_tokenizer(
        tokenizer_name='yttm', tokenizer_model=decoder_tokenizer_model, bpe_dropout=args.bpe_dropout
    )

    tokens_in_batch = [int(item) for item in args.tokens_in_batch.split(',')]
    for num_tokens in tokens_in_batch:
        dataset = TranslationDataset(
            dataset_src=str(Path(args.src_fname).expanduser()),
            dataset_tgt=str(Path(args.tgt_fname).expanduser()),
            tokens_in_batch=num_tokens,
            clean=args.clean,
            max_seq_length=args.max_seq_length,
            min_seq_length=args.min_seq_length,
            max_seq_length_diff=args.max_seq_length,
            max_seq_length_ratio=args.max_seq_length,
            cache_ids=False,
            cache_data_per_node=False,
            use_cache=False,
        )
        print('Batchifying ...')
        dataset.batchify(encoder_tokenizer, decoder_tokenizer)
        start = time.time()
        pickle.dump(dataset, open(os.path.join(args.out_dir, 'batches.tokens.%d.pkl' % (num_tokens)), 'wb'))
        end = time.time()
        print('Took %f time to pickle' % (end - start))
        start = time.time()
        dataset = pickle.load(open(os.path.join(args.out_dir, 'batches.tokens.%d.pkl' % (num_tokens)), 'rb'))
        end = time.time()
        print('Took %f time to unpickle' % (end - start))
