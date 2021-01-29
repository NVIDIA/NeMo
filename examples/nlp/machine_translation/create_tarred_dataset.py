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
import json
import os
import pickle
import tarfile
import tempfile

import youtokentome as yttm

from nemo.collections.nlp.data import TranslationDataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer


def write_batches_to_tarfiles(
    args,
    src_fname,
    tgt_fname,
    num_tokens,
    encoder_tokenizer,
    decoder_tokenizer,
    num_files_in_tar,
    tar_file_ptr,
    tar_file_ctr,
    global_batch_ctr,
):
    """
    Writes current fragment of the overall parallel corpus to tarfiles by:
    (1) Creating a minibatches using a TranslationDataset object.
    (2) Writing each minibatch to a pickle file.
    (3) Adding pickle files to a tarfile until it reaches args.num_batches_per_tarfile.
    """

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
    dataset.batchify(encoder_tokenizer, decoder_tokenizer)

    for _, batch in dataset.batches.items():
        global_batch_ctr += 1
        pickle.dump(batch, open(os.path.join(args.out_dir, 'batch-%d.pkl' % (global_batch_ctr)), 'wb'))

        if num_files_in_tar == args.num_batches_per_tarfile:
            tar_file_ctr += 1
            tar_file_ptr.close()
            tar_file_ptr = tarfile.open(
                os.path.join(args.out_dir, 'batches.tokens.%d.%d.tar' % (num_tokens, tar_file_ctr)), 'w'
            )
            num_files_in_tar = 0

        tar_file_ptr.add(os.path.join(args.out_dir, 'batch-%d.pkl' % (global_batch_ctr)))
        num_files_in_tar += 1
        os.remove(os.path.join(args.out_dir, 'batch-%d.pkl' % (global_batch_ctr)))
    return tar_file_ptr, global_batch_ctr, num_files_in_tar, tar_file_ctr


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
    parser.add_argument('--tokens_in_batch', type=int, default=16000, help='# Tokens per batch per GPU')
    parser.add_argument(
        '--lines_per_dataset_fragment',
        type=int,
        default=1000000,
        help='Number of lines to consider for bucketing and padding',
    )
    parser.add_argument(
        '--num_batches_per_tarfile',
        type=int,
        default=1000,
        help='Number of batches (pickle files) within each tarfile',
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
        os.remove('/tmp/concat_dataset.txt')
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
    tar_file_ctr = 1
    num_files_in_tar = 0
    num_lines = 0
    shard_num = 0
    global_batch_ctr = 0
    tmp_f_src = tempfile.NamedTemporaryFile(delete=False, mode='w')
    tmp_f_tgt = tempfile.NamedTemporaryFile(delete=False, mode='w')
    tar_file_ptr = tarfile.open(os.path.join(args.out_dir, 'batches.tokens.%d.%d.tar' % (tokens_in_batch, 1)), 'w')
    with open(args.src_fname, 'r') as f_src, open(args.tgt_fname) as f_tgt:
        for src_line, tgt_line in zip(f_src, f_tgt):
            tmp_f_src.write(src_line)
            tmp_f_tgt.write(tgt_line)
            num_lines += 1

            if num_lines == args.lines_per_dataset_fragment:
                tmp_f_src.close()
                tmp_f_tgt.close()
                tar_file_ptr, global_batch_ctr, num_files_in_tar, tar_file_ctr = write_batches_to_tarfiles(
                    args,
                    tmp_f_src.name,
                    tmp_f_tgt.name,
                    tokens_in_batch,
                    encoder_tokenizer,
                    decoder_tokenizer,
                    num_files_in_tar=num_files_in_tar,
                    tar_file_ptr=tar_file_ptr,
                    tar_file_ctr=tar_file_ctr,
                    global_batch_ctr=global_batch_ctr,
                )

                num_lines = 0
                shard_num += 1

                os.remove(tmp_f_src.name)
                os.remove(tmp_f_tgt.name)

                tmp_f_src = tempfile.NamedTemporaryFile(delete=False, mode='w')
                tmp_f_tgt = tempfile.NamedTemporaryFile(delete=False, mode='w')

    tmp_f_src.close()
    tmp_f_tgt.close()
    tar_file_ptr, global_batch_ctr, num_files_in_tar, tar_file_ctr = write_batches_to_tarfiles(
        args,
        tmp_f_src.name,
        tmp_f_tgt.name,
        tokens_in_batch,
        encoder_tokenizer,
        decoder_tokenizer,
        num_files_in_tar=num_files_in_tar,
        tar_file_ptr=tar_file_ptr,
        tar_file_ctr=tar_file_ctr,
        global_batch_ctr=global_batch_ctr,
    )
    tar_file_ptr.close()
    os.remove(tmp_f_src.name)
    os.remove(tmp_f_tgt.name)

    if num_files_in_tar != args.num_batches_per_tarfile:
        os.remove(os.path.join(args.out_dir, 'batches.tokens.%d.%d.tar' % (tokens_in_batch, tar_file_ctr)))
        global_batch_ctr -= num_files_in_tar
        print('Dropping %d batches because of overflow' % (num_files_in_tar))

    json.dump({'num_batches': global_batch_ctr}, open(os.path.join(args.out_dir, 'metadata.json'), 'w'))
