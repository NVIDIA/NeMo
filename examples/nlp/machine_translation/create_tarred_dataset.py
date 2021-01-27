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
import tempfile
import tarfile
import numpy as np
import youtokentome as yttm
import math

from nemo.collections.nlp.data import TranslationDataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer

def create_pickled_dataset(args, src_fname, tgt_fname, num_tokens, encoder_tokenizer, decoder_tokenizer, tar_file_ctr, global_batch_ctr):
    dataset = TranslationDataset(
        dataset_src=src_fname,
        dataset_tgt=tgt_fname,
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
    total_batches = len(dataset)
    num_batches_per_tarfile = math.ceil(total_batches / args.num_tar_files_per_dataset_fragment)
    cur_batches = 0
    local_tar_file_ctr = 1
    tar_file_ctr += 1
    f_tar = tarfile.open(os.path.join(
        args.out_dir,
        'batches.tokens.%d.%d.tar' % (num_tokens, tar_file_ctr)
    ), 'w')

    for idx, (_, batch) in enumerate(dataset.batches.items()):
        global_batch_ctr += 1
        pickle.dump(batch, open(os.path.join(
            args.out_dir,
            'batch.%d.pkl' % (global_batch_ctr)
        ), 'wb'))
        f_tar.add(os.path.join(args.out_dir, 'batch.%d.pkl' % (global_batch_ctr)))
        os.remove(os.path.join(args.out_dir, 'batch.%d.pkl' % (global_batch_ctr)))
        cur_batches += 1
        if (
            cur_batches == num_batches_per_tarfile and
            idx != len(dataset.batches) - 1
        ):
            f_tar.close()
            cur_batches = 0
            tar_file_ctr += 1
            local_tar_file_ctr += 1
            print('Creating local tar file %d' % (local_tar_file_ctr))
            f_tar = tarfile.open(os.path.join(
                args.out_dir,
                'batches.tokens.%d.%d.tar' % (num_tokens, tar_file_ctr)
            ), 'w')
    f_tar.close()
    return tar_file_ctr, global_batch_ctr

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
        '--tokens_in_batch', type=int, default=16000, help='# Tokens per batch per GPU'
    )
    parser.add_argument(
        '--lines_per_dataset_fragment', type=int, default=1000000, help='# Tokens per dataset fragment'
    )
    parser.add_argument(
        '--num_tar_files_per_dataset_fragment', type=int, default=4, help='Number of tarfiles'
    )

    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
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

    tokens_in_batch = args.tokens_in_batch
    tar_file_ctr = 0
    num_lines = 0
    shard_num = 0
    global_batch_ctr = 0
    tmp_f_src = tempfile.NamedTemporaryFile(delete=False, mode='w')
    tmp_f_tgt = tempfile.NamedTemporaryFile(delete=False, mode='w')
    with open(args.src_fname, 'r') as f_src, open(args.tgt_fname) as f_tgt:
        for src_line, tgt_line in zip(f_src, f_tgt):
            tmp_f_src.write(src_line)
            tmp_f_tgt.write(tgt_line)
            num_lines += 1
            if num_lines == args.lines_per_dataset_fragment:
                print('Creating dataset shard %d ...' % (shard_num))
                tmp_f_src.close()
                tmp_f_tgt.close()
                tar_file_ctr, global_batch_ctr = create_pickled_dataset(
                    args,
                    tmp_f_src.name,
                    tmp_f_tgt.name,
                    tokens_in_batch,
                    encoder_tokenizer,
                    decoder_tokenizer,
                    tar_file_ctr=tar_file_ctr,
                    global_batch_ctr=global_batch_ctr
                )
                num_lines = 0
                shard_num += 1
                os.unlink(tmp_f_src.name)
                os.unlink(tmp_f_tgt.name)
                tmp_f_src = tempfile.NamedTemporaryFile(delete=False, mode='w')
                tmp_f_tgt = tempfile.NamedTemporaryFile(delete=False, mode='w')

    tmp_f_src.close()
    tmp_f_tgt.close()
    _, _ = create_pickled_dataset(
        args,
        tmp_f_src.name,
        tmp_f_tgt.name,
        tokens_in_batch,
        encoder_tokenizer,
        decoder_tokenizer,
        tar_file_ctr=tar_file_ctr,
        global_batch_ctr=global_batch_ctr
    )
