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
import re
import random
from argparse import ArgumentParser

import numpy as np
import tqdm
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader

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
    for i, abin in enumerate(bins):
        if sum(abin) + s <= bin_size:
            return i
    return -1


def first_fit(seq_lens, max_seq_length):
    res = []
    for s in seq_lens:
        first_bin = find_first_bin_that_fits(res, s, max_seq_length)
        if first_bin == -1:  # open a new bin
            res.append([s])
        else:
            res[first_bin].append(s)
    return res


def first_fit_decreasing(seq_lens, max_seq_length):
    sorted_seq_lens = sorted(seq_lens, reverse=True)
    return first_fit(sorted_seq_lens, max_seq_length)


def first_fit_shuffle(seq_lens, max_seq_length):
    shuffled_seq_lens = seq_lens[:]
    np.random.shuffle(shuffled_seq_lens)
    return first_fit(shuffled_seq_lens, max_seq_length)


def optimized_first_fit(seq_lens, max_seq_length):
    # Convert sequence lengths to a numpy array for efficient computation
    seq_lens_np = np.array(seq_lens)

    # Initialize an array to keep track of the remaining space in each bin
    bins_remaining = np.array([], dtype=int)

    # Iterate over each sequence length
    for s in seq_lens_np:
        # Find the first bin that can fit the sequence
        valid_bins = bins_remaining >= s
        if valid_bins.any():
            # Find the index of the first bin that fits
            first_bin_index = np.where(valid_bins)[0][0]
            # Update the remaining space in that bin
            bins_remaining[first_bin_index] -= s
        else:
            # If no existing bin can fit the sequence, create a new bin
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

def create_assignment(output_path, assignments, ifile_handles):
    n_samples_in_this_shard = len(assignments)
    input_ids, loss_mask, seq_start_id = {}, {}, {}

    for oindex, assignment in tqdm(enumerate(assignments), total=n_samples_in_this_shard):
        _input_ids, _loss_mask, _seq_start_id = [], [], [0]

        for seq_length in assignment:
            _input_ids.extend(ifile_handles[seq_length][0].pop())
            _loss_mask.extend(ifile_handles[seq_length][1].pop())
            _seq_start_id.append(len(_input_ids))

        input_ids[oindex] = _input_ids
        loss_mask[oindex] = _loss_mask
        seq_start_id[oindex] = _seq_start_id[:-1]

    output_data = []
    for i in range(len(input_ids)):
        item_dict = {
            'input_ids': input_ids[i],
            'loss_mask': loss_mask[i],
            'seq_start_id': seq_start_id[i]
        }
        output_data.append(item_dict)

    assert all(not seq[0] for seq in ifile_handles.values()), "Error: There are items left over from the assignment"
    assert all(not seq[1] for seq in ifile_handles.values()), "Error: There are items left over from the assignment"
    np.save(output_path, output_data)
    print("Done, output written to", output_path)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--image_folder", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument("--max_seq_length", default=2048, type=int)
    parser.add_argument('--packing_algorithm', default='first_fit_decreasing', type=str, choices=PACKING_ALGOS)
    parser.add_argument('--seed', default=0, type=int,
                        help="Seed for shuffling, only used if packing_algorithm=first_fit_shuffle")
    parser.add_argument(
        "--hparams_file",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), '../conf/llava_config.yaml'
        ),
        required=False,
        help="Path config for restoring. It's created during training and may need to be modified during restore if restore environment is different than training. Ex: /raid/nemo_experiments/megatron_gpt/hparams.yaml",
    )
    args = parser.parse_args()
    return args


def pack_sequence(args, seq_lens):
    np.random.seed(args.seed)
    random.seed(args.seed)

    # packing_fn = globals()[args.packing_algorithm]
    packing_fn = optimized_first_fit
    assignments = packing_fn(seq_lens, args.max_seq_length)
    return assignments


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
    # for item_dict in tqdm.tqdm(train_dl):
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
    pattern = r'abcabc_seqlen_(\d+).bin'
    seq_len_list = []
    for file in files:
        match = re.match(pattern, file)
        if match:
            seq_len = int(match.group(1))
            seq_len_list.append(seq_len)

    print(seq_len_list)
    aggregated_seq_lens = []
    doc_pop_order = {}
    indexed_datasets = {}
    for seq_len in seq_len_list:
        dataset = IndexedDataset(f"{prefix_path}_seqlen_{seq_len}", multimodal=True)
        aggregated_seq_lens.extend([seq_len] * (len(dataset.document_indices) - 1))
        doc_pop_order[seq_len] = list(np.random.permutation(len(dataset.document_indices) - 1))
        indexed_datasets[seq_len] = dataset

    print("getting assignments")
    assignments = pack_sequence(args, aggregated_seq_lens)
    print("finish getting assignemnets")

    final_builder = IndexedDatasetBuilder(get_bin_path(f"{prefix_path}_packed"), dtype=np.float32, multimodal=True)
    for assignment in tqdm.tqdm(assignments):
        packed_items = collections.defaultdict(list)
        packed_items["seq_indices"] = [0]
        for seq_len in assignment:
            doc_index = doc_pop_order[seq_len].pop()
            doc_start = indexed_datasets[seq_len].document_indices[doc_index]
            doc_end = indexed_datasets[seq_len].document_indices[doc_index + 1]
            item_dict = {"tokens": (indexed_datasets[seq_len][doc_start: doc_end][0])[0],
                         "labels": (indexed_datasets[seq_len][doc_start: doc_end][0])[1],
                         "image": (indexed_datasets[seq_len][doc_start: doc_end][0])[2]}
            for key in ["tokens", "labels", "image"]:
                packed_items[key].append(item_dict[key])
            packed_items["seq_indices"].append(packed_items["seq_indices"][-1] + len(item_dict["tokens"]))

        for key in ["seq_indices", "tokens", "labels", "image"]:
            final_builder.add_item(torch.tensor(np.array(packed_items[key])), 1 if key == "image" else 0)
        final_builder.end_document()
    final_builder.finalize(get_idx_path(f"{prefix_path}_packed"))

    dataset = IndexedDataset(f"{prefix_path}_packed", multimodal=True)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()

# python /opt/NeMo/examples/multimodal/multimodal_llm/neva/sequence_packing/preprocess_dataset.py  --data_path=/lustre/fsw/coreai_dlalgo_genai/datasets/LLaVA-Pretrain-LCS-558K/test.json --image_folder=/lustre/fsw/coreai_dlalgo_genai/datasets/LLaVA-Pretrain-LCS-558K/images  --tokenizer_path=/lustre/fsw/coreai_dlalgo_genai/datasets/checkpoints/tokenizer_add_special.model
