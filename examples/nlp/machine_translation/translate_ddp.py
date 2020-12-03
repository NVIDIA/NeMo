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


import os
from argparse import ArgumentParser

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nemo.collections.nlp.data.machine_translation import TranslationOneSideDataset
from nemo.collections.nlp.models.machine_translation import TransformerMTModel
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.utils import logging


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="")
    parser.add_argument("--text2translate", type=str, required=True, help="")
    parser.add_argument("--tokenizer_model", type=str, required=True, help="")
    parser.add_argument("--max_num_tokens_in_batch", type=int, required=True, help="")
    parser.add_argument("--result_dir", type=str, required=True, help="")
    args = parser.parse_args()
    return args


def setup(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def translate(rank, world_size, args):
    setup(rank, world_size, args)
    ddp_model = TransformerMTModel.load_from_checkpoint(args.model)
    ddp_model.teacher_forcing_forward = False
    ddp_model = DDP(ddp_model.to(rank), device_ids=[rank])
    ddp_model.eval()
    src_tokenizer = get_tokenizer(tokenizer_name='yttm', tokenizer_model=args.tokenizer_model)
    tgt_tokenizer = src_tokenizer
    dataset = TranslationOneSideDataset(
        src_tokenizer,
        args.text2translate,
        tokens_in_batch=args.max_num_tokens_in_batch,
        max_seq_length=2048,
        cache_ids=True,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=False)
    result_dir = os.path.join(args.result_dir, f'rank{rank}')
    os.makedirs(result_dir, exist_ok=True)
    originals_file_name = os.path.join(result_dir, 'originals.txt')
    translations_file_name = os.path.join(result_dir, 'translations.txt')
    num_translated_sentences = 0

    with open(originals_file_name, 'w') as of, open(translations_file_name, 'w') as tf:
        for batch_idx, batch in enumerate(loader):
            for i in range(len(batch)):
                if batch[i].ndim == 3:
                    batch[i] = batch[i].squeeze(dim=0)
                batch[i] = batch[i].to(rank)
            src_ids, src_mask, sent_ids = batch
            if batch_idx % 100 == 0:
                logging.info(
                    f"{batch_idx} batches and {num_translated_sentences} sentences were translated by process with "
                    f"rank {rank}"
                )
            num_translated_sentences += len(src_ids)
            _, translations = ddp_model(src_ids, src_mask)
            translations = translations.cpu().numpy()
            for t in translations:
                tf.write(tgt_tokenizer.ids_to_text(t) + '\n')
            for o in src_ids:
                of.write(src_tokenizer.ids_to_text(o) + '\n')
    cleanup()


def main() -> None:
    world_size = torch.cuda.device_count()
    args = get_args()
    mp.spawn(translate, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
