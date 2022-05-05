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

"""RETRO Style dataset."""

import torch
from nemo.collections.nlp.data.language_modeling.megatron.megatron_dataset import MegatronDataset

# from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import (
#     create_masked_lm_predictions,
#     create_tokens_and_tokentypes,
#     get_a_and_b_segments,
#     get_samples_mapping,
#     truncate_segments,
# )
# from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import MMapIndexedDataset


class MockRETRODataset(MegatronDataset):
    def __init__(
        self, cfg, trainer, tokenizer, name, data_prefix, documents, indexed_dataset, num_samples, seq_length, seed,
    ):
        super().__init__(cfg, trainer=trainer)
        self.name = name
        self._cfg = cfg

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return 10000

    def __getitem__(self, idx):
        vocab_size = self.tokenizer.vocab_size

        neighbors = self._cfg.data.neighbors
        dim = self._cfg.data.retrival_dim
        input_length = self._cfg.encoder_seq_length
        chunks = input_length // dim
        chunk_size = self._cfg.chunk_size
        pad_id = self.eos_id

        all_tokens = torch.randint(0, vocab_size, (input_length + 1))
        hidden = all_tokens[:, :-1]
        labels = all_tokens[:, 1:]

        hidden_mask = (hidden != pad_id)
        retrieved = torch.randint(0, vocab_size, (chunks, neighbors, 2 * chunk_size))

        context_mask = (retrieved != pad_id)
        retrieved_emb = torch.rand(chunks, neighbors, 2 * chunk_size, dim)

        return {
            'tokens': hidden,
            'labels': labels,
            'tokens_mask': hidden_mask,
            'loss_mask': hidden_mask,
            'retrieved_emb_mask': context_mask,
            'retrieved_emb': retrieved_emb,
        }