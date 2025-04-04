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
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from datasets import DatasetDict, load_dataset

from nemo.collections.llm.gpt.data.core import get_dataset_root
from nemo.collections.llm.gpt.data.fine_tuning import HFFineTuningDataModule
from nemo.lightning.io.mixin import IOMixin
from nemo.utils import logging

if TYPE_CHECKING:
    from nemo.collections.common.tokenizers import TokenizerSpec
    from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs


class SquadDataModule(HFFineTuningDataModule):
    """A data module for fine-tuning on the Squad dataset.

    This class inherits from the `HFFineTuningDataModule` class including arguments for init and these methods.
    """

    def _preprocess_and_split_data(
        self, dset: DatasetDict, split_val_from_train: bool = True, val_proportion: float = 0.05
    ):
        """Preprocesses and splits the downloaded dataset into training, validation, and test sets.

        Args:
            dset (DatasetDict): The downloaded dataset object.
            split_val_from_train (bool, optional): Whether to split the validation set from the training set.
                If False, the validation set is split from the test set. Defaults to True.
            val_proportion (float, optional): The proportion of the training or test set to be used
                for the validation split. Defaults to 0.05.
        """
        super()._preprocess_and_split_data(
            dset,
            split_val_from_train=split_val_from_train,
            val_proportion=val_proportion,
        )

    def _make_splits(self, dset, split_val_from_train, val_proportion, *args, **kwargs):
        """Maps train/validation/test to standard split names."""
        save_splits = {}
        train_set = dset.get('train')
        val_set = dset.get('validation')

        if split_val_from_train:
            split_dataset = train_set.train_test_split(test_size=val_proportion, seed=self.seed)
            save_splits['training'] = split_dataset['train']
            save_splits['validation'] = split_dataset['test']
            save_splits['test'] = val_set
        else:
            split_dataset = val_set.train_test_split(test_size=val_proportion, seed=self.seed)
            save_splits['training'] = train_set
            save_splits['validation'] = split_dataset['test']
            save_splits['test'] = split_dataset['train']
        return save_splits

    def _json_line_from_example(self, example, split_name, *args, **kwargs):
        """Extract data for QA task."""
        json_line = {
            "input": "Context: " + example["context"] + " Question: " + example['question'] + " Answer:",
            "output": example["answers"]["text"][0],
        }
        if split_name == "test":
            json_line["original_answers"] = example["answers"]["text"]
        return json_line

    @property
    def dataset_name(self) -> str:
        return "squad"

    def reconfigure_limit_batches(self):
        """no op"""
        return
