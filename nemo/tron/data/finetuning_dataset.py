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

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch

from nemo.collections.llm.gpt.data.core import create_sft_dataset
from nemo.tron.tokenizers.tokenizer import _HuggingFaceTokenizer
from nemo.tron.utils.common_utils import get_rank_safe, print_rank_0

logger = logging.getLogger(__name__)


class FinetuningDatasetBuilder:
    """Builder class for fine-tuning datasets.

    This class provides methods to build datasets for fine-tuning large language models.
    It follows a builder pattern similar to BlendedMegatronDatasetBuilder but adapted for
    fine-tuning scenarios.

    Args:
        dataset_root (Union[str, Path]): The root directory containing training, validation, and test data.
        tokenizer: The tokenizer to use for preprocessing text.
        seq_length (int, optional): The maximum sequence length. Defaults to 2048.
        seed (int, optional): Random seed for data shuffling. Defaults to 1234.
        memmap_workers (int, optional): Number of worker processes for memmap datasets. Defaults to 1.
        max_train_samples (int, optional): Maximum number of training samples. Defaults to None.
        packed_sequence_specs (Optional[dict], optional): Specifications for packed sequences. Defaults to None.
        dataset_kwargs (Optional[dict[str, Any]], optional): Additional dataset creation arguments. Defaults to None.
    """

    def __init__(
        self,
        dataset_root: Union[str, Path],
        tokenizer,
        seq_length: int = 2048,
        seed: int = 1234,
        memmap_workers: int = 1,
        max_train_samples: Optional[int] = None,
        packed_sequence_specs: Optional[dict] = None,
        dataset_kwargs: Optional[dict[str, Any]] = None,
        do_validation: bool = True,
        do_test: bool = True,
    ):
        self.dataset_root = Path(dataset_root)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.seed = seed
        self.memmap_workers = memmap_workers
        self.max_train_samples = max_train_samples
        self.packed_sequence_specs = packed_sequence_specs
        self.packed_sequence_size = (
            -1 if not packed_sequence_specs else packed_sequence_specs.get("packed_sequence_size", -1)
        )
        self.dataset_kwargs = dataset_kwargs or {}
        self._pad_cu_seqlens = (
            False if not packed_sequence_specs else packed_sequence_specs.get("pad_cu_seqlens", False)
        )

        self.do_validation = do_validation
        self.do_test = do_test

        print_rank_0(f"Building FinetuningDatasetBuilder with root={self.dataset_root}")

        if self.packed_sequence_size > 0:
            print_rank_0(f"Using packed sequences with size {self.packed_sequence_size}")

    def prepare_data(self) -> None:
        """Prepare data if needed."""
        self.prepare_packed_data()

    def prepare_packed_data(self) -> None:
        """Prepare packed sequence data if needed."""
        if self.packed_sequence_size > 0:
            from nemo.collections.llm.gpt.data.packed_sequence import prepare_packed_sequence_data

            if not self.train_path_packed.is_file():
                print_rank_0(f"Preparing packed training data at {self.train_path_packed}")
                prepare_packed_sequence_data(
                    input_path=self.train_path,
                    output_path=self.train_path_packed,
                    packed_sequence_size=self.packed_sequence_size,
                    tokenizer=self.tokenizer,
                    max_seq_length=self.seq_length,
                    seed=self.seed,
                    output_metadata_path=self.pack_metadata,
                )

            if not self.validation_path_packed.is_file():
                print_rank_0(f"Preparing packed validation data at {self.validation_path_packed}")
                prepare_packed_sequence_data(
                    input_path=self.validation_path,
                    output_path=self.validation_path_packed,
                    packed_sequence_size=self.packed_sequence_size,
                    tokenizer=self.tokenizer,
                    max_seq_length=self.seq_length,
                    seed=self.seed,
                    output_metadata_path=self.pack_metadata,
                )

    def build(self) -> list[Optional[Any]]:
        """Build train, validation, and test datasets.

        This method creates the necessary datasets based on the configuration.
        It first prepares packed data if needed, then builds the datasets in parallel
        on the appropriate ranks.

        Returns:
            list[Optional[Any]]: A list containing the train, validation, and test datasets.
                                Any of these may be None if not available or not built on current rank.
        """
        # Prepare packed data if needed
        if get_rank_safe() == 0:
            self.prepare_data()

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # This needs to be called on all ranks
        datasets = self._build_datasets()
        return datasets

    def _build_datasets(self) -> list[Optional[Any]]:
        """Internal method to build all datasets.

        Returns:
            list[Optional[Any]]: The train, validation, and test datasets.
        """
        train_ds = self._create_dataset(
            self.train_path if self.packed_sequence_size <= 0 else self.train_path_packed,
            pack_metadata_path=None if self.packed_sequence_size <= 0 else self.pack_metadata,
            max_num_samples=self.max_train_samples,
            **self.dataset_kwargs,
        )

        if self.do_validation:
            valid_ds = self._create_dataset(
                self.validation_path if self.packed_sequence_size <= 0 else self.validation_path_packed,
                pack_metadata_path=None if self.packed_sequence_size <= 0 else self.pack_metadata,
                is_test=True,
                **self.dataset_kwargs,
            )
        else:
            valid_ds = None

        if self.do_test:
            test_ds = self._create_dataset(
                self.test_path,
                is_test=True,
                **self.dataset_kwargs,
            )
        else:
            test_ds = None

        return [train_ds, valid_ds, test_ds]

    @lru_cache
    def _create_dataset(self, path, pack_metadata_path=None, is_test=False, **kwargs):
        """Create a dataset from the given path and parameters.

        Args:
            path: Path to the dataset file
            pack_metadata_path: Path to the packed sequence metadata
            is_test: Whether this is a test dataset
            **kwargs: Additional arguments to pass to the dataset constructor

        Returns:
            The created dataset
        """
        if not Path(path).exists():
            print_rank_0(f"Warning: Dataset path {path} does not exist")
            return None

        is_not_packing = self.packed_sequence_size <= 0
        return create_sft_dataset(
            path,
            tokenizer=self.tokenizer,
            seq_length=(self.seq_length if is_not_packing else self.packed_sequence_size),
            memmap_workers=self.memmap_workers,
            seed=self.seed,
            is_test=is_test,
            pack_metadata_file_path=None if is_not_packing else pack_metadata_path,
            pad_cu_seqlens=False if is_not_packing else self._pad_cu_seqlens,
            **kwargs,
        )

    @property
    def train_path(self) -> Path:
        """Path to training dataset file"""
        return self.dataset_root / "training.jsonl"

    @property
    def default_pack_path(self) -> Path:
        """The default directory to write packing files."""
        tokenizer_model_name = self._extract_tokenizer_model_name()
        default_pack_path = self.dataset_root / "packed" / tokenizer_model_name
        if not default_pack_path.exists():
            default_pack_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using default path for packing files: {str(default_pack_path)}")

        return default_pack_path

    @property
    def pack_metadata(self) -> Path:
        """Path to metadata dataset file for packed sequence."""
        if self.packed_sequence_size > 0:
            if self.packed_sequence_specs.get("packed_metadata_path") is not None:
                return self.packed_sequence_specs["packed_metadata_path"]
            return self.default_pack_path / f"{self.packed_sequence_size}_metadata.jsonl"
        else:
            raise ValueError("pack_metadata invalid since packed sequence size is not specified.")

    @property
    def train_path_packed(self) -> Path:
        """Path to training dataset file for packed sequence."""
        if self.packed_sequence_size > 0:
            if self.packed_sequence_specs.get("packed_train_data_path") is not None:
                return self.packed_sequence_specs["packed_train_data_path"]
            return self.default_pack_path / f"training_{self.packed_sequence_size}.npy"
        else:
            raise ValueError("`train_path_packed` invalid since packed sequence size is not specified.")

    @property
    def validation_path_packed(self) -> Path:
        """Path to validation dataset file for packed sequence."""
        if self.packed_sequence_size > 0:
            if self.packed_sequence_specs.get("packed_val_data_path") is not None:
                return self.packed_sequence_specs["packed_val_data_path"]
            return self.default_pack_path / f"validation_{self.packed_sequence_size}.npy"
        else:
            raise ValueError("`validation_path_packed` invalid since packed sequence size is not specified.")

    @property
    def validation_path(self) -> Path:
        """Path to validation dataset file"""
        return self.dataset_root / "validation.jsonl"

    @property
    def test_path(self) -> Path:
        """Path to test dataset file"""
        return self.dataset_root / "test.jsonl"

    def _extract_tokenizer_model_name(self) -> str:
        """Automatically get the model name from model path."""
        if self.packed_sequence_specs and self.packed_sequence_specs.get("tokenizer_model_name") is not None:
            return self.packed_sequence_specs["tokenizer_model_name"]
        elif isinstance(self.tokenizer, _HuggingFaceTokenizer):
            name = self.tokenizer._tokenizer.name_or_path
            if name.endswith("context/nemo_tokenizer"):
                # NEMO_HOME/hf_org/hf_model/context/nemo_tokenizer => hf_org--hf_model
                tokenizer_model_name = "--".join(name.split("/")[-4:-2])
            elif name.endswith("nemo_tokenizer"):
                # NEMO_HOME/hf_org/hf_model/nemo_tokenizer => hf_org--hf_model
                tokenizer_model_name = "--".join(name.split("/")[-3:-1])
            else:
                # hf_org/hf_model => hf_org--hf_model
                tokenizer_model_name = name.replace("/", "--")
            return tokenizer_model_name
        else:
            return f"unknown_tokenizer_{hash(self.tokenizer)}"
