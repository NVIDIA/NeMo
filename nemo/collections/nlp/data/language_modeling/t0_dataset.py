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
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec

import numpy as np
import torch

from nemo.collections.nlp.data.language_modeling.text_memmap_dataset import JSONLMemMapDataset, TextMemMapDataset
from nemo.collections.nlp.data.language_modeling.megatron.dataset_utils import get_samples_mapping, make_text_memmap_bin_compatibility
from nemo.core.classes import Dataset


class T0Dataset(Dataset):

    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerSpec,
        max_src_seq_length: int = 512,
        max_tgt_seq_length: int = 512,
        replace_bos_with_pad: bool = False,
        add_bos_to_input: bool = False,
        add_eos_to_input: bool = False,
        max_num_samples: int = None,
        seed: int = 1234
    ):
        """
        src_file_name: Path to a JSONL T0 dataset file.
        tokenizer: Tokenizer for the dataset. Instance of a class that inherits TokenizerSpec (ex: YTTM, SentencePiece).
        max_src_seq_length: Maximum length of the source sequences. Lines above this length will be truncated.
        max_tgt_seq_length: Maximum length of the target sequences. Lines above this length will be truncated.
        replace_bos_with_pad: Whether the decoder starts with a pad token. This is needed for Google's T5 models that may be converted from HF.
        add_bos_to_input: Whether to add the bos_id to the input sequence.
        add_eos_to_input: Whether to add the eos_id to the input sequence.
        seed: Random seed for data shuffling.
        max_num_samples: Maximum number of samples to load. This can be > dataset length if you want to oversample data. If None, all samples will be loaded.
        """
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.max_src_seq_length = max_src_seq_length
        self.max_tgt_seq_length = max_tgt_seq_length
        self.replace_bos_with_pad = replace_bos_with_pad
        self.add_bos_to_input = add_bos_to_input
        self.add_eos_to_input = add_eos_to_input
        self.max_num_samples = max_num_samples
        self.seed = seed

        self.indexed_dataset = JSONLMemMapDataset(dataset_paths=[file_path], tokenizer=None)

        # Create compatibility with Megatron samples mapping
        if self.max_num_samples is not None:
            make_text_memmap_bin_compatibility(self.indexed_dataset)

        # Will be None after this call if `max_num_samples` is None
        self._build_samples_mapping()

    def _build_samples_mapping(self):
        if self.max_num_samples is not None:
            # This means max src and max tgt sequence length need to be the same
            if self.max_src_seq_length != self.max_tgt_seq_length:
                raise ValueError(
                    f"max_src_seq_length ({self.max_src_seq_length}) != max_tgt_seq_length ({self.max_tgt_seq_length}). This is needed for max_samples based training for now."
                )

            self.samples_mapping = get_samples_mapping(
                indexed_dataset=self.indexed_dataset,
                data_prefix=self.file_path,
                num_epochs=None,
                max_num_samples=self.max_num_samples,
                max_seq_length=self.max_src_seq_length - 2,
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
        Process a single example from the dataset into IDs and other T0-related metadata.
        """
        tokenized_input = self.tokenizer.text_to_ids(example['input'])
        tokenized_output = self.tokenizer.text_to_ids(example['output'])
        offset = 0
        if self.add_bos_to_input:
            offset += 1
        if self.add_eos_to_input:
            offset += 1
        if len(tokenized_input) > self.max_src_seq_length - offset:
            tokenized_input = tokenized_input[: self.max_src_seq_length - 2]
        if len(tokenized_output) > self.max_tgt_seq_length - 2:
            tokenized_output = tokenized_output[: self.max_tgt_seq_length - 2]
        bos_id = self.tokenizer.pad_id if self.replace_bos_with_pad else self.tokenizer.bos_id
        if self.add_bos_to_input:
            tokenized_input = [bos_id] + tokenized_input
        if self.add_eos_to_input:
            tokenized_input = tokenized_input + [self.tokenizer.eos_id]
        target = [bos_id] + tokenized_output + [self.tokenizer.eos_id]
        original = ""
        template = ""
        for item in example['chunked_idx'].split(', '):
            item = item.split('-')
            if item[0] == "original_text":
                original += example['input'][int(item[1]) : int(item[2])]
            elif item[0] == "template":
                template += example['input'][int(item[1]) : int(item[2])]
            else:
                raise ValueError(f"Unknown chunk type: {item[0]}")

        return {
            'text_enc': tokenized_input,
            'text_dec': target[:-1],
            'labels': target[1:],
            'original': self.tokenizer.text_to_ids(original),
            'template': self.tokenizer.text_to_ids(template),
            'prompt': self.tokenizer.text_to_ids(example['prompt']),
        }

    def _maybe_cast_to_list(self, x):
        if isinstance(x, np.ndarray):
            return [item.tolist() for item in x]
        return x

    def collate_fn(self, batch):
        enc_query = [item['text_enc'] for item in batch]
        dec_input = [item['text_dec'] for item in batch]
        labels = [item['labels'] for item in batch]
        original = [item['original'] for item in batch]
        template = [item['template'] for item in batch]
        prompt = [item['prompt'] for item in batch]

        enc_query = self._maybe_cast_to_list(enc_query)
        dec_input = self._maybe_cast_to_list(dec_input)
        labels = self._maybe_cast_to_list(labels)
        original = self._maybe_cast_to_list(original)
        template = self._maybe_cast_to_list(template)
        prompt = self._maybe_cast_to_list(prompt)

        max_dec_input_length = max([len(item) for item in dec_input]) if dec_input else 0
        max_enc_query_length = max([len(item) for item in enc_query]) if enc_query else 0
        max_label_length = max([len(item) for item in labels]) if labels else 0
        max_original_length = max([len(item) for item in original]) if original else 0
        max_template_length = max([len(item) for item in template]) if template else 0
        max_prompt_length = max([len(item) for item in prompt]) if prompt else 0

        loss_mask = [([1] * (len(item))) + ([0] * (max_label_length - len(item))) for item in labels]
        enc_query = [item + [self.tokenizer.pad_id] * (max_enc_query_length - len(item)) for item in enc_query]
        dec_input = [item + [self.tokenizer.pad_id] * (max_dec_input_length - len(item)) for item in dec_input]
        labels = [item + [self.tokenizer.pad_id] * (max_label_length - len(item)) for item in labels]
        original = [item + [self.tokenizer.pad_id] * (max_original_length - len(item)) for item in original]
        template = [item + [self.tokenizer.pad_id] * (max_template_length - len(item)) for item in template]
        prompt = [item + [self.tokenizer.pad_id] * (max_prompt_length - len(item)) for item in prompt]

        enc_query = torch.LongTensor(enc_query)
        dec_input = torch.LongTensor(dec_input)
        labels = torch.LongTensor(labels)
        loss_mask = torch.LongTensor(loss_mask)
        original = torch.LongTensor(original)
        template = torch.LongTensor(template)
        prompt = torch.LongTensor(prompt)

        enc_mask = (enc_query != self.tokenizer.pad_id).long()
        dec_mask = (dec_input != self.tokenizer.pad_id).long()

        return {
            'text_enc': enc_query,
            'text_dec': dec_input,
            'labels': labels,
            'loss_mask': loss_mask,
            'enc_mask': enc_mask,
            'dec_mask': dec_mask,
            'original': original,
            'template': template,
            'prompt': prompt,
        }
