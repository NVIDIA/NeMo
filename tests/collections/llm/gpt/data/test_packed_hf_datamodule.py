# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import MagicMock

import pytest
import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader

from nemo.collections import llm
from nemo.collections.llm.gpt.data.hf_dataset import (
    SquadHFDataModule,
    batchify,
    extract_key_from_dicts,
    make_dataset_splits,
    pad_within_micro,
)

SQUAD_HF_CACHE = "/home/TestData/lite/hf_cache/squad/"
SQUAD_NEMO_CACHE = "/home/TestData/lite/nemo_cache/squad"


def test_collate_fn_packed():
    # Instantiate HFDatasetDataModulePacked class
    dm = llm.HFDatasetDataModulePacked(
        path_or_dataset=Dataset.from_dict({"dummy": [0]}),
        split="train",
        packed_sequence_size=5,  # Must match or exceed test sequence lengths
    )

    batch = [
        {"id": [1], "token_ids": [1, 2, 3], "seq_lens": 3},
        {"id": [2], "token_ids": [123], "seq_lens": 1}
    ]
    result = dm.collate_fn(batch)
    print('result= ' + str(result))
    assert isinstance(result, dict)
    assert "attention_mask" in result  # New packed feature
    assert result["attention_mask"].shape == (1, 1, 4, 4), result["attention_mask"].shape  # Mask matches packed_sequence_size
    assert torch.all(
        result["attention_mask"] == torch.tensor([[[[ True, False, False, False],
          [ True,  True, False, False],
          [ True,  True,  True, False],
          [False, False, False,  True]]]])
    )


    # Verify inherited collation behavior
    assert "id" in result
    assert "token_ids" in result
    assert result["token_ids"].shape == (1, 4)  # Padded to packed_sequence_size
    assert torch.all(result["token_ids"][1] == torch.LongTensor([123, 0, 0]))

