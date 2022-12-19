# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import torch

from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import get_samples_mapping
from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import JSONLMemMapDataset
from nemo.core.classes import Dataset

__all__ = ['GPTSFTDataset']


class GPTSFTDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerSpec,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        add_sep: bool = False,
        sep_id: int = None,
        max_num_samples: int = None,
        seed: int = 1234,
    ):
        """
        file_path: Path to a JSONL GPT supervised fine-tuning dataset.
        tokenizer: Tokenizer for the dataset. Instance of a class that inherits TokenizerSpec (ex: YTTM, SentencePiece).
        max_seq_length (int): maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset. Data examples will be dropped if they do not meet the min length requirements. 
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add an end of sentence token to each data example
        add_sep (bool): Whether to add a separation token to each data example (goes between prompt and answer)
        tokens_to_generate (int): (inference only) Number of tokens to generate during inference
        seed: Random seed for data shuffling.
        max_num_samples: Maximum number of samples to load. This can be > dataset length if you want to oversample data. If None, all samples will be loaded.
        seed: int = 1234,
        """
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.add_sep = add_sep
        self.sep_id = sep_id
        self.max_num_samples = max_num_samples
        self.seed = seed

        self.indexed_dataset = JSONLMemMapDataset(dataset_paths=[file_path], tokenizer=None, header_lines=0)

        # Will be None after this call if `max_num_samples` is None
        self._build_samples_mapping()

    def _build_samples_mapping(self):
        if self.max_num_samples is not None:
            #TODO: double-check this for decoder-only GPT
            self.samples_mapping = get_samples_mapping(
                indexed_dataset=self.indexed_dataset,
                data_prefix=self.file_path,
                num_epochs=None,
                max_num_samples=self.max_num_samples,
                max_seq_length=self.max_seq_length - 2,
                short_seq_prob=0,
                seed=self.seed,
                name=self.file_path.split('/')[-1],
                binary_head=False,
            )
        else:
            self.samples_mapping = None

    def __len__(self):
        if self.max_num_samples is None:
            return len(self.indexed_dataset)
        else:
            return len(self.samples_mapping)

    def __getitem__(self, idx):
        if isinstance(idx, np.int64):
            idx = idx.item()

        if self.samples_mapping is not None:
            assert idx < len(self.samples_mapping)
            idx, _, _ = self.samples_mapping[idx]
            if isinstance(idx, np.uint32):
                idx = idx.item()

        assert idx < len(self.indexed_dataset)
        example = self.indexed_dataset[idx]
        return self._process_example(example)

    def _process_example(self, example):
        """
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.
        """

        text_ids = self.tokenizer.text_to_ids(example['text'])
        answer_ids = self.tokenizer.text_to_ids(example['answer'])

        total_ids = len(text_ids) + len(answer_ids)
        if self.add_bos:
            total_ids += 1
        if self.add_sep:
            total_ids += 1
        if self.add_eos:
            total_ids += 1

        # If the total number of token is greater than the max, we will try to truncate the answer
        if total_ids > self.max_seq_length:
            truncation_length = total_ids - self.max_seq_length
            answer_ids = answer_ids[: -min(truncation_length, len(answer_ids))]

        input_ids = text_ids

        # Adds sep token between text/prompt and answer
        if self.add_sep:
            input_ids = input_ids + [self.sep_id]

        answer_start_idx = len(input_ids)

        input_ids = input_ids + answer_ids

        if self.add_bos:
            input_ids = [self.tokenizer.bos_id] + input_ids
        if self.add_eos:
            input_ids = input_ids + [self.tokenizer.eos_id]

        # We return None if the example fails in the min_seq_length and max_seq_length checks
        # TODO: verify on the dataloader how to skip Nones and keep constant batch size
        # Right now this is handled offline before the training step
        if len(input_ids) < self.min_seq_length or len(input_ids) > self.max_seq_length:
            input_ids= input_ids[:self.max_seq_length]
            #return None

        processed_example = {
            'input_ids' : input_ids,
            'answer_start_idx' : answer_start_idx,
        }

        return processed_example

    def _maybe_cast_to_list(self, x):
        if isinstance(x, np.ndarray):
            return [item.tolist() for item in x]
        return x

    def _collate_item(self, item):
        item = self._maybe_cast_to_list(item)
        #max_length = max([len(x) for x in item]) if item else 0
        # here [0] should be tokenizer.pad_id
        item = [x + [0] * (self.max_seq_length - len(x)) for x in item]
        return item

    def _build_loss_mask(self, processed_example):
        """ Pad input_ids in batch to max batch length while building loss mask """
        input_ids = processed_example['input_ids']
        answer_start_idx = processed_example['answer_start_idx']

        loss_mask = [float(idx >= answer_start_idx) for idx in range(len(input_ids))]

        return loss_mask

    @torch.no_grad()
    def _create_attention_mask(self):
        """Create `attention_mask`.
        Args:
            input_ids: A 1D tensor that holds the indices of tokens.
        """
        #seq_length = len(input_ids)
        # `attention_mask` has the shape of [1, seq_length, seq_length]
        attention_mask = torch.tril(torch.ones((self.max_seq_length, self.max_seq_length))).unsqueeze(0)
        attention_mask = attention_mask < 0.5
        return attention_mask

    def collate_fn(self, batch):
        """
        """ 

        input_ids = [item['input_ids'][:-1] for item in batch]
        labels = [item['input_ids'][1:] for item in batch]
        loss_mask = [self._build_loss_mask(item)[1:] for item in batch]
        attention_mask = [self._create_attention_mask() for _ in batch]
        attention_mask = torch.stack(attention_mask)

        position_ids = [list(range(self.max_seq_length)) * len(batch)]
        position_ids = torch.LongTensor(position_ids)
        input_ids = torch.LongTensor(self._collate_item(input_ids))
        labels = torch.LongTensor(self._collate_item(labels))
        loss_mask = torch.LongTensor(self._collate_item(loss_mask))

        processed_batch = {
            'tokens' : input_ids,
            'labels' : labels,
            'attention_mask' : attention_mask,
            'loss_mask': loss_mask,
            'position_ids' : position_ids,
        }

        return processed_batch
        