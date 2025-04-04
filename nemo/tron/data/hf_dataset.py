# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import glob
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, TypedDict, Union, cast

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

from nemo.collections.llm.gpt.data.core import get_dataset_root
from nemo.tron.config import FinetuningDatasetConfig
from nemo.tron.data.finetuning_dataset import FinetuningDatasetBuilder
from nemo.tron.tokenizers.tokenizer import MegatronTokenizer
from nemo.tron.utils.common_utils import print_rank_0

logger = logging.getLogger(__name__)


class ProcessExampleOutput(TypedDict):
    input: str
    output: str
    original_answers: list[str]


class ProcessExampleFn(Protocol):
    def __call__(
        self, example: dict[str, Any], tokenizer: Optional[MegatronTokenizer] = None
    ) -> ProcessExampleOutput: ...


@dataclass(kw_only=True)
class HFDatasetConfig(FinetuningDatasetConfig):
    dataset_name: str
    process_example_fn: ProcessExampleFn
    dataset_subset: Optional[str] = None
    dataset_dict: Optional[DatasetDict] = None
    split: Optional[str] = None
    download_mode: Optional[str] = None
    val_proportion: Optional[float] = 0.05
    split_val_from_train: bool = True
    delete_raw: bool = False
    hf_kwargs: Optional[dict[str, Any]] = None
    hf_filter_lambda: Optional[Callable] = None
    hf_filter_lambda_kwargs: Optional[dict[str, Any]] = None


def preprocess_and_split_data(
    dset: DatasetDict,
    dataset_name: str,
    dataset_root: Path,
    tokenizer: MegatronTokenizer,
    process_example_fn: ProcessExampleFn,
    split_val_from_train: bool = True,
    val_proportion: Optional[float] = None,
    train_aliases: tuple[str] = ("train", "training"),
    test_aliases: tuple[str] = ("test", "testing"),
    val_aliases: tuple[str] = ("val", "validation", "valid", "eval"),
    delete_raw: bool = False,
    seed: int = 1234,
    rewrite: bool = False,
):
    """Preprocesses and splits the downloaded dataset into training, validation, and test sets."""
    logger.info(f"Preprocessing {dataset_name} to jsonl format and splitting...")
    save_splits = {}
    train_set: Dataset | None = None
    val_set: Dataset | None = None
    test_set: Dataset | None = None

    for alias in train_aliases:
        train_set = dset.get(alias)
        if train_set is not None:
            break

    for alias in val_aliases:
        val_set = dset.get(alias)
        if val_set is not None:
            break

    for alias in test_aliases:
        test_set = dset.get(alias)
        if test_set is not None:
            break

    assert train_set, f"Train set with aliases: {train_aliases} not found in dataset"
    train_set = cast(Dataset, train_set)

    if val_proportion:
        if split_val_from_train:
            split_dataset = train_set.train_test_split(test_size=val_proportion, seed=seed)
            save_splits["training"] = split_dataset["train"]
            save_splits["validation"] = split_dataset["test"]
            if val_set:
                save_splits["test"] = val_set
        else:
            assert val_set, f"Validation set with aliases: {val_aliases} not found in dataset"
            val_set = cast(Dataset, val_set)
            split_dataset = val_set.train_test_split(test_size=val_proportion, seed=seed)
            save_splits["training"] = train_set
            save_splits["validation"] = split_dataset["test"]
            save_splits["test"] = split_dataset["train"]
    else:
        save_splits["training"] = train_set
        if val_set:
            save_splits["validation"] = val_set
        if test_set:
            save_splits["test"] = test_set

    if test_set:
        test_set = cast(Dataset, test_set)
        save_splits["test"] = test_set

    for split_name, dataset in save_splits.items():
        output_file = dataset_root / f"{split_name}.jsonl"

        if output_file.exists() and output_file.is_file():
            if not rewrite:
                logger.info(f"{output_file} exists, skipping...")
                continue
            else:
                logger.info(f"{output_file} exists, deleting and rewriting...")
                os.remove(output_file)
                for p in glob.glob(str(output_file) + "*"):
                    os.remove(p)

        with output_file.open("w", encoding="utf-8") as f:
            for example in tqdm(dataset, desc=f"Processing {split_name} split"):
                json_line = {}

                processed_example = process_example_fn(example, tokenizer)
                # Write each example as a JSON line in the output file
                json_line["input"] = processed_example["input"]
                json_line["output"] = processed_example["output"]
                if split_name == "test":
                    json_line["original_answers"] = processed_example["original_answers"]
                f.write(json.dumps(json_line) + "\n")

        logger.info(f"{split_name} split saved to {output_file}")

    if delete_raw:
        for p in dataset_root.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
            elif ".jsonl" not in str(p.name):
                p.unlink()


class HFDatasetBuilder(FinetuningDatasetBuilder):
    """Builder class for Hugging Face datasets.

    This class extends FinetuningDatasetBuilder to work with Hugging Face datasets instead of file paths.
    It provides methods to build datasets from Hugging Face's datasets library.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        is_built_on_rank: Callable,
        process_example_fn: ProcessExampleFn,
        dataset_dict: Optional[DatasetDict] = None,
        dataset_subset: Optional[str] = None,
        dataset_root: Optional[Union[str, Path]] = None,
        split=None,
        seq_length=1024,
        seed: int = 1234,
        memmap_workers: int = 1,
        max_train_samples: Optional[int] = None,
        packed_sequence_specs: Optional[dict] = None,
        download_mode: Optional[str] = None,
        val_proportion: Optional[float] = 0.05,
        split_val_from_train: bool = True,
        delete_raw: bool = False,
        hf_kwargs: Optional[dict[str, Any]] = None,
        dataset_kwargs: Optional[dict[str, Any]] = None,
        hf_filter_lambda: Optional[Callable] = None,
        hf_filter_lambda_kwargs: Optional[dict[str, Any]] = None,
        do_validation: bool = True,
        do_test: bool = True,
    ) -> None:
        dataset_root = Path(dataset_root) if dataset_root else get_dataset_root(dataset_name)

        # Initialize the parent class with common parameters
        super().__init__(
            dataset_root=dataset_root,
            tokenizer=tokenizer,
            seq_length=seq_length,
            seed=seed,
            memmap_workers=memmap_workers,
            is_built_on_rank=is_built_on_rank,
            dataset_kwargs=dataset_kwargs,
            max_train_samples=max_train_samples,
            packed_sequence_specs=packed_sequence_specs,
            do_validation=do_validation,
            do_test=do_test,
        )

        # HF-specific attributes
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.dataset_dict = dataset_dict
        self.split = split
        self.download_mode = download_mode
        self.hf_kwargs = hf_kwargs or {}
        self.val_proportion = val_proportion
        self.split_val_from_train = split_val_from_train
        self.delete_raw = delete_raw
        self.process_example_fn = process_example_fn
        self.hf_filter_lambda = hf_filter_lambda
        self.hf_filter_lambda_kwargs = hf_filter_lambda_kwargs or {}

        if not val_proportion:
            self.do_validation = False
            self.do_test = False

        print_rank_0(f"Building HFDataset {self.dataset_name}")

    def prepare_data(self) -> None:
        if self.download_mode != "force_redownload" and self.hf_filter_lambda:
            raise ValueError("`hf_filter_lambda` is not supported when `download_mode` is not `force_redownload`")

        if self.dataset_dict:
            dataset = self.dataset_dict
        else:
            dataset = self._load_dataset()

        if self.hf_filter_lambda:
            dataset = dataset.filter(self.hf_filter_lambda, **self.hf_filter_lambda_kwargs)

        preprocess_and_split_data(
            dataset,
            self.dataset_name,
            self.dataset_root,
            tokenizer=self.tokenizer,
            process_example_fn=self.process_example_fn,
            split_val_from_train=self.split_val_from_train,
            val_proportion=self.val_proportion,
            delete_raw=self.delete_raw,
            seed=self.seed,
            rewrite=True,
        )
        super().prepare_data()

    def _load_dataset(self):
        """Load the dataset from Hugging Face or use the provided dataset."""
        if isinstance(self.dataset_name, str):
            logger.info(f"Loading HF dataset from {self.dataset_name} to {self.dataset_root}")
            dataset = load_dataset(
                self.dataset_name,
                name=self.dataset_subset,
                cache_dir=str(self.dataset_root),
                split=self.split,
                **self.hf_kwargs,
                download_mode=self.download_mode,
            )
        else:
            raise ValueError("Expected `dataset_name` to be str, got " + str(type(self.dataset_name)))

        return dataset
