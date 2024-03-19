# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import collections
import os
import random
import re

import numpy as np
import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from megatron.core.datasets.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)
from nemo.collections.multimodal.data.neva.neva_dataset import (
    make_supervised_data_module,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

PACKING_ALGOS = ['first_fit_decreasing', 'first_fit_shuffle']


def find_first_bin_that_fits(bins, s, bin_size):
    """
    Finds the first bin that can fit an item of size s.

    Parameters:
    - bins: List of bins where each bin is a list of item sizes.
    - s: Size of the current item.
    - bin_size: Maximum capacity of each bin.

    Returns:
    - Index of the first bin that can fit the item, or -1 if none can.
    """
    for i, abin in enumerate(bins):
        if sum(abin) + s <= bin_size:
            return i
    return -1


def first_fit(seq_lens, max_seq_length):
    """
    Assigns sequences to bins using the First Fit algorithm.

    Parameters:
    - seq_lens: List of sequence lengths.
    - max_seq_length: Maximum capacity of each bin.

    Returns:
    - List of bins with assigned sequence lengths.
    """
    res = []
    for s in seq_lens:
        first_bin = find_first_bin_that_fits(res, s, max_seq_length)
        if first_bin == -1:  # open a new bin
            res.append([s])
        else:
            res[first_bin].append(s)
    return res


def first_fit_decreasing(seq_lens, max_seq_length):
    """
    Assigns sequences to bins using the First Fit Decreasing algorithm.

    Parameters:
    - seq_lens: List of sequence lengths.
    - max_seq_length: Maximum capacity of each bin.

    Returns:
    - List of bins with assigned sequence lengths.
    """
    sorted_seq_lens = sorted(seq_lens, reverse=True)
    return first_fit(sorted_seq_lens, max_seq_length)


def first_fit_shuffle(seq_lens, max_seq_length):
    """
    Assigns sequences to bins using a shuffled version of the First Fit algorithm.

    Parameters:
    - seq_lens: List of sequence lengths.
    - max_seq_length: Maximum capacity of each bin.

    Returns:
    - List of bins with assigned sequence lengths.
    """
    shuffled_seq_lens = seq_lens[:]
    np.random.shuffle(shuffled_seq_lens)
    return first_fit(shuffled_seq_lens, max_seq_length)


def shuffle_and_pack(seq_lens, max_seq_length):
    """
    Assigns sequences to bins with shuffling, trying to maximize the packing efficiency.

    Parameters:
    - seq_lens: List of sequence lengths.
    - max_seq_length: Maximum capacity of each bin.

    Returns:
    - List of bins with assigned sequence lengths.
    """
    shuffled_seq_lens = np.array(seq_lens)
    np.random.shuffle(shuffled_seq_lens)
    bins = [[]]
    cur_bin_total = 0
    for s in tqdm(shuffled_seq_lens):
        if cur_bin_total + s <= max_seq_length:
            bins[-1].append(s)
            cur_bin_total += s
        else:
            bins.append([s])
            cur_bin_total = s
    return bins


def optimized_first_fit(seq_lens, max_seq_length):
    """
    An optimized version of the first fit algorithm using numpy for efficiency.

    Parameters:
    - seq_lens: List of sequence lengths.
    - max_seq_length: Maximum capacity of each bin.

    Returns:
    - List of bins with assigned sequence lengths, optimized for space usage.
    """
    seq_lens_np = np.array(seq_lens)
    bins_remaining = np.array([], dtype=int)
    for s in seq_lens_np:
        valid_bins = bins_remaining >= s
        if valid_bins.any():
            first_bin_index = np.where(valid_bins)[0][0]
            bins_remaining[first_bin_index] -= s
        else:
            bins_remaining = np.append(bins_remaining, max_seq_length - s)

    # Calculate the final bins from the bins_remaining information
    # This part is mainly for reconstructing the final bins structure similar to the original function's output
    bins = [[] for _ in range(len(bins_remaining))]
    for s in seq_lens:
        for i, space in enumerate(bins_remaining + s):
            if space >= max_seq_length:
                bins[i].append(s)
                bins_remaining[i] -= s
                break
    return bins


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--image_folder", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument("--max_seq_length", default=2048, type=int)
    parser.add_argument('--packing_algorithm', default='first_fit_decreasing', choices=PACKING_ALGOS, type=str)
    parser.add_argument('--seed', default=0, type=int, help="Seed for shuffling, used with first_fit_shuffle.")
    parser.add_argument("--hparams_file", type=str, default=os.path.join(os.path.dirname(__file__), '../conf/llava_config.yaml'), required=False, help="Path to the hparams file.")
    return parser.parse_args()

def pack_sequence(args, seq_lens):
    """
    Packs sequences according to the specified algorithm in args.

    Parameters:
    - args: Command line arguments.
    - seq_lens: List of sequence lengths.

    Returns:
    - List of bins with assigned sequence lengths.
    """
    np.random.seed(args.seed)
    random.seed(args.seed)

    # packing_fn = globals()[args.packing_algorithm]
    packing_fn = shuffle_and_pack
    bins = packing_fn(seq_lens, args.max_seq_length)
    return bins


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = get_args()
    nemo_config = OmegaConf.load(args.hparams_file)
    nemo_config.model.data.data_path = args.data_path
    nemo_config.model.data.image_folder = args.image_folder

    tokenizer = get_nmt_tokenizer(
        library="sentencepiece",
        tokenizer_model=args.tokenizer_path,
    )
    train_ds = make_supervised_data_module(tokenizer=tokenizer, model_cfg=nemo_config.model)["train_dataset"]
    train_dl = DataLoader(train_ds, num_workers=32, collate_fn=None, shuffle=False)
    # Example shape: {'tokens': torch.Size([1, 344]), 'labels': torch.Size([1, 344]), 'image': torch.Size([1, 1, 3, 224, 224])}
    prefix_path = "abcabc"
    # builders = {}
    # for item_dict in tqdm(train_dl):
    #     item_dict = {k: v[0] for k, v in item_dict.items()}
    #     seq_len = len(item_dict['tokens'])
    #     if seq_len in builders:
    #         builder = builders[seq_len]
    #     else:
    #         builder = IndexedDatasetBuilder(get_bin_path(f"{prefix_path}_seqlen_{seq_len}"), dtype=np.float32, multimodal=True)
    #         builders[seq_len] = builder
    #     builder.add_item(item_dict['tokens'])
    #     builder.add_item(item_dict['labels'])
    #     builder.add_item(item_dict['image'], 1)
    #     builder.end_document()
    #     del item_dict
    # for seq_len, builder in builders.items():
    #     builder.finalize(get_idx_path(f"{prefix_path}_seqlen_{seq_len}"))

    files = os.listdir('.')
    pattern = rf"{prefix_path}_seqlen_(\d+).bin"
    seq_len_list = []
    for file in files:
        match = re.match(pattern, file)
        if match:
            seq_len = int(match.group(1))
            seq_len_list.append(seq_len)

    aggregated_seq_lens = []
    doc_pop_order = {}
    indexed_datasets = {}
    for seq_len in seq_len_list:
        dataset = IndexedDataset(f"{prefix_path}_seqlen_{seq_len}", multimodal=True)
        aggregated_seq_lens.extend([seq_len] * (len(dataset.document_indices) - 1))
        doc_pop_order[seq_len] = list(np.random.permutation(len(dataset.document_indices) - 1))
        indexed_datasets[seq_len] = dataset

    print("getting bins")
    bins = pack_sequence(args, aggregated_seq_lens)
    print("finish getting bins")

    final_builder = IndexedDatasetBuilder(get_bin_path(f"{prefix_path}"), dtype=np.float32, multimodal=True)
    for assignment in tqdm(bins):
        packed_items = collections.defaultdict(list)
        packed_items["seq_indices"] = [0]
        for seq_len in assignment:
            doc_index = doc_pop_order[seq_len].pop()
            doc_start = indexed_datasets[seq_len].document_indices[doc_index]
            doc_end = indexed_datasets[seq_len].document_indices[doc_index + 1]
            item_dict = {"tokens": torch.tensor((indexed_datasets[seq_len][doc_start: doc_end][0])[0]),
                         "labels": torch.tensor((indexed_datasets[seq_len][doc_start: doc_end][0])[1]),
                         "image": torch.tensor((indexed_datasets[seq_len][doc_start: doc_end][0])[2]),
                         }
            for key in ["tokens", "labels", "image"]:
                packed_items[key].append(item_dict[key])
            packed_items["seq_indices"].append(packed_items["seq_indices"][-1] + seq_len)

        for key in ["seq_indices", "tokens", "labels", "image"]:
            final_builder.add_item(
                torch.tensor(packed_items[key]) if key == "seq_indices" else torch.cat(packed_items[key], dim=0),
                1 if key == "image" else 0,
            )
        final_builder.end_document()
    final_builder.finalize(get_idx_path(f"{prefix_path}"))

if __name__ == '__main__':
    main()

# python /opt/NeMo/examples/multimodal/multimodal_llm/neva/sequence_packing/preprocess_dataset.py  --data_path=/lustre/fsw/coreai_dlalgo_genai/datasets/LLaVA-Pretrain-LCS-558K/test.json --image_folder=/lustre/fsw/coreai_dlalgo_genai/datasets/LLaVA-Pretrain-LCS-558K/images  --tokenizer_path=/lustre/fsw/coreai_dlalgo_genai/datasets/checkpoints/tokenizer_add_special.model
