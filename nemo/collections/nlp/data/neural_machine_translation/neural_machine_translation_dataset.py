# Copyright (c) 2020, MeetKai Inc.  All rights reserved.
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
from typing import Dict, List, Optional, Tuple

import numpy as np

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.core.classes import Dataset
from nemo.core.neural_types import ChannelType, MaskType, NeuralType

__all__ = ["NeuralMachineTranslationDataset"]


class NeuralMachineTranslationDataset(Dataset):
    """A dataset class that converts raw data to a dataset that can be used by NeuralMachineTranslationModel.

    Args:
        filepath: .tsv file to sequence + label.
            the first line is header (sentence [tab] label)
            each line should be [sentence][tab][label]
        encoder_tokenizer: encoder tokenizer object such as AutoTokenizer
        decoder_tokenizer: decoder tokenizer object. If using BART or end to end model, set this to encoder_tokenizer
        max_seq_length: max sequence length including bos and eos tokens
        num_samples: number of samples you want to use for the dataset. If -1, use all dataset. Useful for testing.
        convert_labels: if true, converts labels for masked lm and updates pad_id to -100
            for hf masked loss
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
               """
        return {
            "input_ids": NeuralType(("B", "T"), ChannelType()),
            "attention_mask": NeuralType(("B", "T"), MaskType()),
            "decoder_input_ids": NeuralType(("B", "T"), ChannelType()),
            "lm_labels": NeuralType(("B", "T"), ChannelType()),
        }

    def __init__(
        self,
        filepath: str,
        encoder_tokenizer: AutoTokenizer,
        decoder_tokenizer: AutoTokenizer,
        encoder_add_special_tokens: bool,
        decoder_add_special_tokens: bool,
        max_seq_length: int,
        num_samples: int = -1,
        convert_labels: bool = False,
    ):
        self.filepath = filepath
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.encoder_add_special_tokens = encoder_add_special_tokens
        self.decoder_add_special_tokens = decoder_add_special_tokens
        self.max_seq_length = max_seq_length
        self.num_samples = num_samples
        self.convert_labels = convert_labels

        if num_samples == 0:
            raise ValueError("num_samples has to be positive.", num_samples)

        if self.max_seq_length and self.max_seq_length <= 2:
            self.max_seq_length = None

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"{filepath} not found. The filepath must be set in train_ds.filepath and validation_ds.filepath."
            )

        with open(filepath) as f:
            lines = f.readlines()[1:]

        if num_samples > 0:
            lines = lines[:num_samples]

        input_ids, input_masks, label_ids = [], [], []
        for line in lines:
            try:
                sentence, label = line.split("\t")
            except ValueError:
                raise ValueError("Each line of input file should contain the format [sentence][tab][label].")
            ids, mask = self.text_to_ids(
                sentence, tokenizer=encoder_tokenizer, add_special_tokens=encoder_add_special_tokens
            )
            input_ids.append(ids)
            input_masks.append(mask)
            label_ids.append(
                self.text_to_ids(label, tokenizer=decoder_tokenizer, add_special_tokens=decoder_add_special_tokens)[0]
            )

        self.input_ids = np.asarray(input_ids)
        self.input_masks = np.asarray(input_masks)
        self.label_ids = np.asarray(label_ids)

    def text_to_ids(
        self, text: str, tokenizer: AutoTokenizer, add_special_tokens=False
    ) -> Tuple[List[int], List[int]]:
        """Converts text to ids. Truncates and adds padding."""
        text_tokens = tokenizer.text_to_ids(text.strip())
        num_special_tokens = 2 if add_special_tokens else 0

        if self.max_seq_length and self.max_seq_length > num_special_tokens:
            text_tokens = text_tokens[: self.max_seq_length - num_special_tokens]
        if add_special_tokens:
            text_tokens = [tokenizer.bos_id] + text_tokens + [tokenizer.eos_id]
        mask = [1] * len(text_tokens)

        if self.max_seq_length and self.max_seq_length > num_special_tokens:
            pad_length = self.max_seq_length - len(text_tokens)
            text_tokens += [tokenizer.pad_id] * pad_length
            mask += [0] * pad_length

        return text_tokens, mask

    def __len__(self):
        return len(self.input_ids)

    def convert_label_ids(self, label_ids: List[int]) -> Tuple[List[int], List[int]]:
        decoder_input_ids = label_ids[:-1]
        lm_labels = label_ids[1:].copy()
        lm_labels[label_ids[1:] == self.decoder_tokenizer.pad_id] = -100  # for huggingface masked lm loss
        return decoder_input_ids, lm_labels

    def __getitem__(self, idx):
        if self.convert_labels:
            decoder_input_ids, lm_labels = self.convert_label_ids(self.label_ids[idx])
        else:
            decoder_input_ids = self.label_ids[idx]
            lm_labels = self.label_ids[idx]
        return self.input_ids[idx], self.input_masks[idx], decoder_input_ids, lm_labels
