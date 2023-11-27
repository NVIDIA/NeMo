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

"""GPT style dataset."""

import os
import time

import numpy as np
import torch
from omegaconf.dictconfig import DictConfig

from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
    get_train_valid_test_split_,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import deallocate_indexed_dataset_memory
from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import make_dataset as make_indexed_dataset
from nemo.core import Dataset
from nemo.utils import logging

try:
    from megatron.core import parallel_state
    from megatron.core.models.retro.data.query.retro_dataset import RetroDataset as MCoreRETRODataset
    from megatron.core.models.retro.data.db.utils import get_merged_train_dataset as get_db_dataset
    from megatron.core.models.retro.data.query.chunk_dataset import get_chunk_dataset_map

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

class RETRODataset(Dataset):
    def __init__(
        self,
        cfg,
        num_neighbors,
        num_retrieved_chunks,
        block_size,
        db_dataset,
        chunk_dataset,
        neighbor_path_map
    ):
        super().__init__()

        self.reset_position_ids = cfg.data.get('reset_position_ids', False)
        self.reset_attention_mask = cfg.data.get('reset_attention_mask', False)
        self.eod_mask_loss = cfg.data.get('eod_mask_loss', False)
        self.eos_id = tokenizer.eos_id

        self.mcore_retro_dataset = MCoreRETRODataset(
                                        num_neighbors,
                                        num_retrieved_chunks,
                                        block_size,
                                        db_dataset,
                                        chunk_dataset,
                                        neighbor_path_map            
                                        )
        return

    def __len__(self):
        return len(self.mcore_retro_dataset.chunk_dataset.sample_dataset)

    def _get_text(self, idx: int):
        n_chunks_per_sample = self.mcore_retro_dataset.chunk_dataset.n_chunks_per_sample

        # Get standard sample.
        sample = self.mcore_retro_dataset.chunk_dataset.sample_dataset[idx]

        # Sample idx to chunk idxs.
        chunk_idxs = list(range(
            idx * n_chunks_per_sample,
            (idx + 1) * n_chunks_per_sample,
        ))

        # Collect retrieved tokens.
        all_retrieved_chunk_ids = []
        all_retrieved_token_ids = []
        for chunk_idx in chunk_idxs:

            # Neighbor chunk ids.
            neighbor_path = self.mcore_retro_dataset.neighbor_path_map[chunk_idx]
            with h5py.File(neighbor_path, "r") as f:
                neighbor_chunk_ids = f["neighbors"] \
                    [chunk_idx % self.mcore_retro_dataset.block_size, :self.mcore_retro_dataset.num_neighbors].tolist()

            # Retrieved (neighbor + continuation) token ids.
            retrieved_chunk_ids = []
            retrieved_token_ids = []
            for neighbor_chunk_id in neighbor_chunk_ids:
                current_chunk_ids = [
                    i % len(self.mcore_retro_dataset.db_dataset)
                    for i in range(
                            neighbor_chunk_id,
                            neighbor_chunk_id + self.mcore_retro_dataset.num_retrieved_chunks)]
                current_token_ids = [self.mcore_retro_dataset.db_dataset[ci]["text"]
                                     for ci in current_chunk_ids]
                retrieved_chunk_ids.append(current_chunk_ids)
                retrieved_token_ids.append(current_token_ids)

            # Collect retrieved tokens.
            all_retrieved_chunk_ids.append(retrieved_chunk_ids)
            all_retrieved_token_ids.append(retrieved_token_ids)

        # Reshape retrieved tokens.
        all_retrieved_chunk_ids = np.array(all_retrieved_chunk_ids) \
            .reshape((n_chunks_per_sample, self.mcore_retro_dataset.num_neighbors, -1))
        all_retrieved_token_ids = np.array(all_retrieved_token_ids) \
            .reshape((n_chunks_per_sample, self.mcore_retro_dataset.num_neighbors, -1))

        # Sample.
        sample = {
            **sample,
            "neighbor_chunks" : all_retrieved_chunk_ids,
            "neighbor_tokens" : all_retrieved_token_ids,
        }

        return sample

    def __getitem__(self, idx):

        sample = self._get_text(idx)

        # Unpack
        tokens_ = sample['text'].long()
        labels = tokens_[1:].contiguous()
        tokens = tokens_[:-1].contiguous()

        # note: [l * k, r]
        # note: 2x == neighbor, continuation
        neighbor_tokens = sample['neighbor_tokens'] \
            .view(-1, retro_args.retro_gpt_retrieved_length).long()

        # Get the masks and postition ids.
        from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
        tokens = torch.unsqueeze(tokens, 0)     # get_ltor_masks_and_position_ids takes as input tokens arguments as a batch (2D tensor)
        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)
        tokens = tokens[0]
        attention_mask = attention_mask[0]
        loss_mask = loss_mask[0]
        position_ids = position_ids[0]
        _, _, neighbor_position_ids = get_ltor_masks_and_position_ids(
            neighbor_tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)
        neighbor_attention_mask = None

        return {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': context_position_idsmask,
            'neighbor_tokens': neighbor_tokens,
            'neighbor_attention_mask': neighbor_attention_mask,
            'neighbor_position_ids': neighbor_position_ids
        }

def build_train_valid_test_datasets(
    cfg,
):

    # convert cfg to args and retro_args
    args = get_args_from_config(cfg)
    import types
    retro_args_path = os.path.join(cfg.get('retro_workdir'), "args.json")
    assert os.path.exists(retro_args_path), "retro workdir missing args.json"
    retro_args = types.SimpleNamespace(**json.load(retro_args_path))

    # DB dataset.
    db_dataset = get_db_dataset(args)   # Lawrence is working on to make it not using global variables

    # Retro datasets.
    chunk_ds_info_map = get_chunk_dataset_map(args) # Lawrence is working on to make it not using global variables
    retro_dataset_map = {}
    for data_key, chunk_ds_info in chunk_ds_info_map.items():
        chunk_dataset = chunk_ds_info["data"]
        neighbor_dir = chunk_ds_info["neighbor_dir"]
        neighbor_path_map = BlockPathMap.from_dir(neighbor_dir, retro_args.retro_block_size)

        # Verify dataset prefixes.
        expected_dir = get_neighbor_dirname(data_key, chunk_dataset.sample_dataset)
        assert expected_dir == neighbor_dir, \
            "inconsistent dataset source; '%s' vs. '%s'." % \
            (expected_dir, neighbor_dir)

        # Verify num chunks.
        n_sample_chunks = len(chunk_dataset)
        n_neighbor_chunks = neighbor_path_map.max_idx

        if not os.path.isdir(neighbor_dir):
            if torch.distributed.get_rank() == 0:
                raise Exception("neighbor directory '%s' not found; please "
                                "compare --train-samples, --seq-length, --seed, "
                                "--eval-iters, and --eval-interval, with "
                                "retro preprocessing args." %
                                neighbor_dir)
            torch.distributed.barrier()
            exit()

        if verify_sizes and n_sample_chunks != n_neighbor_chunks:
            if torch.distributed.get_rank() == 0:
                print("neighbor_dir : %s" % neighbor_dir)
                print("neighbor_path_map : %s" % neighbor_path_map)
                raise Exception("num sampled chunks (%d) != num neighbor chunks "
                                "(%d); did you complete querying the entire "
                                "pretraining dataset?"
                                % (n_sample_chunks, n_neighbor_chunks))
            torch.distributed.barrier()
            exit()

        # Retro dataset.
        retro_dataset_map[data_key] = RetroDataset(
            num_neighbors=args.retro_num_neighbors,
            num_retrieved_chunks=args.retro_num_retrieved_chunks,
            block_size=retro_args.retro_block_size,
            db_dataset=db_dataset,
            chunk_dataset=chunk_dataset,
            neighbor_path_map=neighbor_path_map,
        )

    # Extract datasets.
    train_ds = retro_dataset_map.get("train", None)
    valid_ds = retro_dataset_map.get("valid", None)
    test_ds = retro_dataset_map.get("test", None)

    return train_ds, valid_ds, test_ds