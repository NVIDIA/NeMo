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


import torch
from datasets import Dataset

from nemo.collections import llm
from nemo.collections.llm.gpt.data.hf_dataset import (
    batchify,
    extract_key_from_dicts,
    make_dataset_splits,
    pad_within_micro,
)

from nemo.collections.llm.gpt.data.hf_dataset_packed_sequence import HFDatasetPackedSequenceHelper, CROSS_ENTROPY_IGNORE_IDX

SQUAD_HF_CACHE = "/home/TestData/lite/hf_cache/squad/"
SQUAD_NEMO_CACHE = "/home/TestData/lite/nemo_cache/squad"


def test_collate_fn_packed():
    # Instantiate HFDatasetDataModulePacked class
    dm = llm.HFDatasetDataModulePacked(
        path_or_dataset=Dataset.from_dict({"dummy": [0]}),
        split="train",
        packed_sequence_size=5,  # Must match or exceed test sequence lengths
    )

    batch = {"id": [1, 2], "token_ids": [1, 2, 3, 123], "seq_lens": [3, 1]}

    result = dm.collate_fn(batch)
    print('result= ' + str(result))
    assert isinstance(result, dict)
    assert "attention_mask" in result  # New packed feature
    assert result["attention_mask"].shape == (1, 1, 4, 4), result["attention_mask"].shape
    assert torch.all(
        result["attention_mask"] == torch.tensor([[[[ True, False, False, False],
          [ True,  True, False, False],
          [ True,  True,  True, False],
          [False, False, False,  True]]]])
    )

    # Verify collation behavior
    assert "id" in result
    assert "token_ids" in result
    assert result["token_ids"].shape == (1, 4)  # Padded to packed_sequence_size
    assert torch.all(result["token_ids"] == torch.LongTensor([1, 2, 3, 123]))


@pytest.fixture
def dummy_dataset():
    """Creates a dummy dataset with simple sequential input_ids and labels"""
    samples = [
        {"input_ids": [1, 2, 3], "labels": [10, 20, 30]},
        {"input_ids": [44, 5], "labels": [40, 50]},
        {"input_ids": [66, 7, 8, 9], "labels": [60, 70, 80, 90]},
    ]
    return Dataset.from_list(samples)

def test_pack_sequences(dummy_dataset):
    packed_sequence_size = 6
    max_packs = 2
    split_across_pack = False


    dm = llm.HFDatasetDataModulePacked(
        path_or_dataset=dummy_dataset,
        split="train",
        packed_sequence_size=packed_sequence_size,  # Must match or exceed test sequence lengths
    )

    helper = HFDatasetPackedSequenceHelper(dummy_dataset, split="train")
    packed_dataset = helper.pack(
        packed_sequence_size=packed_sequence_size,
        split_across_pack=split_across_pack,
        max_packs=max_packs
    )
    print(packed_dataset)
    for x in packed_dataset:
        print('x = ' + str(x))
        result = dm.collate_fn(x)
        print('result= ' + str(result))
    # Ensure output is a HuggingFace dataset
    assert isinstance(packed_dataset, Dataset)
    assert len(packed_dataset) == max_packs

    for i in range(max_packs):
        item = packed_dataset[i]
        assert len(item["input_ids"]) == packed_sequence_size
        assert len(item["labels"]) == packed_sequence_size
        assert len(item["position_ids"]) == packed_sequence_size
        assert item["input_ids"][-1] == 0 or isinstance(item["input_ids"][-1], int)  # Padding value
        assert item["labels"][-1] == CROSS_ENTROPY_IGNORE_IDX or isinstance(item["labels"][-1], int)

    # Optional: inspect the structure of seq_lens
    assert all(isinstance(seq_len, int) for pack in packed_dataset["seq_lens"] for seq_len in pack)
