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
import pickle
from argparse import ArgumentParser

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sacremoses import MosesDetokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nemo.collections.nlp.data.machine_translation import (
    TarredOneSideTranslationDataset,
    TarredTranslationDataset,
    TranslationDataset,
    TranslationOneSideDataset,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTEncDecModel
from nemo.collections.nlp.modules.common.tokenizer_utils import get_tokenizer
from nemo.utils import logging


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="")
    parser.add_argument("--text2translate", type=str, required=True, help="")
    parser.add_argument("--result_dir", type=str, required=True, help="")
    parser.add_argument("--twoside", action="store_true")
    parser.add_argument('--metadata_path', type=str, required=True, help="")
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
    torch.set_grad_enabled(False)
    if args.model.endswith(".nemo"):
        logging.info("Attempting to initialize from .nemo file")
        model = MTEncDecModel.restore_from(restore_path=args.model)
    elif args.model.endswith(".ckpt"):
        logging.info("Attempting to initialize from .ckpt file")
        model = MTEncDecModel.load_from_checkpoint(checkpoint_path=args.model)
    model.replace_beam_with_sampling(topk=500)
    ddp_model = DDP(model.to(rank), device_ids=[rank])
    ddp_model.eval()
    if args.twoside:
        assert args.load_from_cached_dataset
        dataset = pickle.load(open(args.text2translate, 'rb'))
    else:
        dataset = TarredOneSideTranslationDataset(
            text_tar_filepaths=args.text2translate,
            metadata_path=args.metadata_path,
            tokenizer=model.encoder_tokenizer,
            shuffle_n=100,
            shard_strategy="scatter",
            world_size=world_size,
            global_rank=rank
        )
    loader = DataLoader(dataset, batch_size=1)
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
            if args.twoside:
                src_ids, src_mask, _, _, _, _ = batch
            else:
                src_ids, src_mask = batch
            if batch_idx % 100 == 0:
                logging.info(
                    f"{batch_idx} batches ({num_translated_sentences} sentences) were translated by process with "
                    f"rank {rank}"
                )
            num_translated_sentences += len(src_ids)
            translations = ddp_model(src_ids, src_mask, None, None)
            translations = translations.cpu().numpy()
            for t in translations:
                tf.write(model.decoder_tokenizer.ids_to_text(t) + '\n')
            for o in src_ids:
                of.write(model.decoder_tokenizer.ids_to_text(o) + '\n')
    cleanup()


def main() -> None:
    world_size = torch.cuda.device_count()
    args = get_args()
    mp.spawn(translate, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
