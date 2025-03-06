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

from nemo.collections import llm
import pytest
import torch
from datasets import Dataset, DatasetDict
from unittest.mock import MagicMock

DATA_PATH = "/home/TestData/lite/hf_cache/squad/"


def test_load_single_split():
    ds = llm.HFDatasetDataModule(
        path_or_dataset=DATA_PATH,
        split='train',
        seq_length=512,
        micro_batch_size=2,
        global_batch_size=2,
    )
    from datasets.arrow_dataset import Dataset

    assert isinstance(ds.dataset_splits, dict)
    assert len(ds.dataset_splits) == 3
    assert 'train' in ds.dataset_splits
    assert ds.dataset_splits['train'] is not None
    assert ds.train is not None
    assert isinstance(ds.dataset_splits['train'], Dataset)
    assert 'val' in ds.dataset_splits
    assert ds.dataset_splits['val'] is None
    assert ds.val is None
    assert 'test' in ds.dataset_splits
    assert ds.dataset_splits['test'] is None
    assert ds.test is None


def test_load_single_split_with_subset():
    ds = llm.HFDatasetDataModule(
        path_or_dataset=DATA_PATH,
        split='train[:10]',
        seq_length=512,
        micro_batch_size=2,
        global_batch_size=2,
    )
    from datasets.arrow_dataset import Dataset

    assert isinstance(ds.dataset_splits, dict)
    assert len(ds.dataset_splits) == 3
    assert 'train' in ds.dataset_splits
    assert ds.dataset_splits['train'] is not None
    assert ds.train is not None
    assert isinstance(ds.dataset_splits['train'], Dataset)
    assert 'val' in ds.dataset_splits
    assert ds.dataset_splits['val'] is None
    assert ds.val is None
    assert 'test' in ds.dataset_splits
    assert ds.dataset_splits['test'] is None
    assert ds.test is None


def test_load_nonexistent_split():
    exception_msg = ''
    expected_msg = '''Unknown split "this_split_name_should_not_exist". Should be one of ['train', 'validation'].'''
    try:
        llm.HFDatasetDataModule(
            path_or_dataset=DATA_PATH,
            split='this_split_name_should_not_exist',
            seq_length=512,
            micro_batch_size=2,
            global_batch_size=2,
        )
    except ValueError as e:
        exception_msg = str(e)
    assert exception_msg == expected_msg, exception_msg


def test_load_multiple_split():
    ds = llm.HFDatasetDataModule(
        path_or_dataset=DATA_PATH,
        split=['train', 'validation'],
        seq_length=512,
        micro_batch_size=2,
        global_batch_size=2,
    )
    from datasets.arrow_dataset import Dataset

    assert isinstance(ds.dataset_splits, dict)
    assert len(ds.dataset_splits) == 3
    assert 'train' in ds.dataset_splits
    assert ds.dataset_splits['train'] is not None
    assert ds.train is not None
    assert isinstance(ds.dataset_splits['train'], Dataset)
    assert isinstance(ds.train, Dataset)
    assert 'val' in ds.dataset_splits
    assert ds.dataset_splits['val'] is not None
    assert ds.val is not None
    assert isinstance(ds.dataset_splits['val'], Dataset)
    assert isinstance(ds.val, Dataset)
    assert 'test' in ds.dataset_splits
    assert ds.dataset_splits['test'] is None
    assert ds.test is None


def test_load_multiple_split_with_subset():
    ds = llm.HFDatasetDataModule(
        path_or_dataset=DATA_PATH,
        split=['train[:100]', 'validation'],
        seq_length=512,
        micro_batch_size=2,
        global_batch_size=2,
    )
    from datasets.arrow_dataset import Dataset

    assert isinstance(ds.dataset_splits, dict)
    assert len(ds.dataset_splits) == 3
    assert 'train' in ds.dataset_splits
    assert ds.dataset_splits['train'] is not None
    assert ds.train is not None
    assert isinstance(ds.dataset_splits['train'], Dataset)
    assert isinstance(ds.train, Dataset)
    assert 'val' in ds.dataset_splits
    assert ds.dataset_splits['val'] is not None
    assert ds.val is not None
    assert isinstance(ds.dataset_splits['val'], Dataset)
    assert isinstance(ds.val, Dataset)
    assert 'test' in ds.dataset_splits
    assert ds.dataset_splits['test'] is None
    assert ds.test is None


def test_validate_dataset_asset_accessibility_file_does_not_exist():
    raised_exception = False
    try:
        llm.HFDatasetDataModule(
            path_or_dataset="/this/path/should/not/exist/",
            seq_length=512,
            micro_batch_size=2,
            global_batch_size=2,
        )
    except FileNotFoundError:
        raised_exception = True

    assert raised_exception == True, "Expected to raise a FileNotFoundError"


def test_validate_dataset_asset_accessibility_file_is_none():
    exception_msg = ''
    expected_msg = "Expected `path_or_dataset` to be str, Dataset, DatasetDict, but got <class 'NoneType'>"
    try:
        llm.HFDatasetDataModule(
            path_or_dataset=None,
            seq_length=512,
            micro_batch_size=2,
            global_batch_size=2,
        )
    except ValueError as e:
        exception_msg = str(e)

    assert exception_msg == expected_msg, exception_msg


@pytest.fixture
def sample_dataset():
    return DatasetDict({
        "train": Dataset.from_dict({
            "id": ["1", "2"],
            "title": ["Title1", "Title2"],
            "context": ["This is a context.", "Another context."],
            "question": ["What is this?", "What about this?"],
            "answers": [{"text": ["A context"]}, {"text": ["Another"]}]
        }),
        "validation": Dataset.from_dict({
            "id": ["3"],
            "title": ["Title3"],
            "context": ["Validation context."],
            "question": ["What is validation?"],
            "answers": [{"text": ["Validation answer"]}]
        })
    })

@pytest.fixture
def data_module(sample_dataset):
    return HFDatasetDataModule(path_or_dataset=sample_dataset, split=["train", "validation"])

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.text_to_ids.side_effect = lambda text: [ord(c) for c in text]  # Mock character-based token IDs
    tokenizer.bos_id = 1
    tokenizer.eos_id = 2
    return tokenizer

@pytest.fixture
def squad_data_module(mock_tokenizer, sample_dataset):
    return SquadHFDataModule(tokenizer=mock_tokenizer, path_or_dataset=sample_dataset, split=["train", "validation"])


def test_dataset_splits(data_module):
    assert data_module.train is not None
    assert data_module.val is not None
    assert data_module.test is None  # No test split in sample dataset


def test_dataloaders(data_module):
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)


def test_formatting_prompts_func(squad_data_module):
    example = {
        "context": "This is a context.",
        "question": "What is this?",
        "answers": {"text": ["A context"]}
    }
    result = squad_data_module.formatting_prompts_func(example)
    
    assert "input_ids" in result
    assert "labels" in result
    assert "loss_mask" in result
    assert len(result["input_ids"]) == len(result["labels"])
