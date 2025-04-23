import os
from pathlib import Path
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import lightning.pytorch as pl
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset

from nemo.utils import logging


def clean_split(name):
    """Removes split from name

    Args:
        name (str): partition name (e.g. "train[:100]")

    Returns:
        str: return partition name without any selector (e.g. "train").
    """
    if "[" in name:
        name = name.split("[")[0]
    if '+' in name:
        name = name.split('+')[0]
    return name


def make_dataset_splits(dataset, split, split_aliases):
    """
    Given a dataset (e.g. from datasets.load_dataset or datasets.Dataset.from_dict) it
    returns a dictionary containing the corresponding dataset splits.

    For example:

    $ ds = load_dataset("dataset-id")
    $ ans = make_dataset_splits(ds)

    # `ds` contains the following
    $ print(ds)
    > DatasetDict({
    >    train: Dataset({
    >        features: ['id', 'title', 'context', 'question', 'answers'],
    >        num_rows: 87599
    >    })
    >    validation: Dataset({
    >        features: ['id', 'title', 'context', 'question', 'answers'],
    >        num_rows: 10570
    >    })
    > })

    # In this case the value of `ans` (returned value) will be:
    $ print(ans)
    > {
    >    "train": Dataset .. (with 87599 rows),
    >    "val": Dataset .. (with 10570 rows),
    > }
    """
    valid_split_names = ["train", "test", "val"]
    dataset_splits = {_split: None for _split in valid_split_names}

    alias_to_split = {}
    for split_name, _split_aliases in split_aliases.items():
        assert split_name in valid_split_names
        for alias in _split_aliases:
            alias_to_split[alias] = split_name
    for name in valid_split_names:
        alias_to_split[name] = name

    if isinstance(dataset, Dataset):
        assert isinstance(split, str), "Expected split to be a string, but got {}".format(type(split))
        split = clean_split(split)
        split = alias_to_split[split]
        dataset_splits[split] = dataset
    elif isinstance(dataset, DatasetDict):
        dataset_split_names = dataset.keys()
        logging.info("HF dataset has the following splits: {}".format(dataset_split_names))
        for alias_split_name, split in dataset.items():
            split_name = alias_to_split[alias_split_name]
            assert dataset_splits[split_name] is None
            dataset_splits[split_name] = split
    elif isinstance(split, list):
        logging.info("Loaded HF dataset will use {} splits.".format(split))
        assert isinstance(dataset, list)
        for i, alias_split_name in enumerate(map(clean_split, split)):
            split_name = alias_to_split[alias_split_name]
            assert dataset_splits[split_name] is None
            dataset_splits[split_name] = dataset[i]
    elif isinstance(split, str):
        logging.info("Loaded HF dataset has a single split.")
        assert not isinstance(dataset, list)
        alias_split_name = split
        if "+" in alias_split_name:
            raise ValueError("Split concatenation not supported")
        elif "[" in alias_split_name:
            alias_split_name = alias_split_name.split("[")[0]
        split_name = alias_to_split[alias_split_name]
        assert dataset_splits[split_name] is None
        dataset_splits[split_name] = dataset
    else:
        raise ValueError("Expected split name to be None, str or a list")

    assert set(valid_split_names) == set(dataset_splits.keys()), dataset_splits.keys()
    num_init_splits = sum(map(lambda x: x is not None, dataset_splits.values()))
    assert num_init_splits > 0, "Expected at least one split to have been initialized {}".format(num_init_splits)
    return dataset_splits


def has_dist_env_init_or_rank_env_var():
    """returns whether it runs on a dist-environment"""
    return dist.is_initialized() or int(os.environ.get('WORLD_SIZE', '0')) > 1


def batchify(tensor):
    """Ensures that the input tensor has at least two dimensions by adding an extra batch dimension if necessary.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to be batchified.

    Returns
    -------
    torch.Tensor
        The tensor with an extra dimension added if it was originally 1-dimensional.
        Otherwise, the tensor is returned as-is.
    """
    if tensor.ndim == 1:
        return tensor.unsqueeze_(0)
    return tensor


def extract_key_from_dicts(batch, key):
    """Extracts the value of the given key from each dictionary in a list of dictionaries.

    Parameters
    ----------
    batch : List[dict]
        A list of dictionaries.
    key : str
        The key whose values are to be extracted from each dictionary.

    Returns
    -------
    List
        A list of values associated with the specified key, in the same order as
        the dictionaries in the input batch.
    """
    return list(map(lambda x: x[key], batch))


def pad_within_micro(batch, pad_token_id, pad_seq_len_divisible=None):
    """Pads each list in a batch of lists to the same length with a specified token.

    Parameters
    ----------
    batch : List[List[int]]
        A batch of sequences (e.g., token IDs), where each sequence is a list of integers.
    pad_token_id : int
        The token ID to use for padding shorter sequences.
    pad_seq_len_divisible : int
        The value to use for padding sequence length so that it is divisible by pad_seq_len_divisible.
    Returns
    -------
    List[List[int]]
        A batch of sequences where each inner list has been padded with the pad token
        to match the length of the longest sequence in the batch.
    """
    max_len = max(map(len, batch))
    if pad_seq_len_divisible:
        max_len = (pad_seq_len_divisible - max_len % pad_seq_len_divisible) + max_len
    return [item + [pad_token_id] * (max_len - len(item)) for item in batch]


class HFDatasetBuilder:
    """A builder class for loading and managing datasets from the Hugging Face datasets library.

    This class provides a builder-like interface for creating datasets and dataloaders
    from Hugging Face datasets. It handles dataset loading, splitting, and creation of
    PyTorch DataLoaders with appropriate collation functions and samplers.

    Args:
        path_or_dataset (str | Dataset | DatasetDict): The dataset name or a preloaded dataset.
        split (str | list, optional): The dataset split(s) to load (e.g., "train" or ["train", "validation"]).
        tokenizer: The tokenizer to use for processing text data.
        seq_length (int, optional): Maximum sequence length for tokenized inputs. Defaults to 1024.
        collate_fn (callable, optional): Custom function for batching data. Defaults to None.
        num_workers (int, optional): Number of workers for data loading. Defaults to 2.
        pin_memory (bool, optional): Whether to use pinned memory for faster GPU transfers. Defaults to True.
        persistent_workers (bool, optional): Whether to keep worker threads alive between epochs. Defaults to True.
        micro_batch_size (int, optional): Batch size per device. Defaults to 2.
        global_batch_size (int, optional): Total batch size across all devices. Defaults to 2.
        pad_token_id (int, optional): Token ID used for padding sequences. Defaults to 0.
        use_dist_sampler (bool, optional): Whether to enable distributed sampling. Defaults to False.
        train_aliases (list, optional): Alternative names for the training split. Defaults to ["train", "training"].
        test_aliases (list, optional): Alternative names for the test split. Defaults to ["test", "testing"].
        val_aliases (list, optional): Alternative names for the validation split.
            Defaults to ["val", "validation", "valid", "eval"].
        pad_seq_len_divisible (int, optional): If provided, pad sequence length to be divisible by this value.
            Defaults to None.
        seed (int, optional): Random seed for shuffling and other random operations. Defaults to 1234.
        do_validation (bool, optional): Whether to prepare validation data. Defaults to True.
        do_test (bool, optional): Whether to prepare test data. Defaults to True.
        num_replicas (int, optional): Number of replicas for distributed sampling. Defaults to None.
        rank (int, optional): Rank for distributed sampling. Defaults to None.
        dataset_kwargs (dict, optional): Additional arguments for dataset creation. Defaults to None.
        **kwargs: Additional arguments passed to `datasets.load_dataset`.
    """

    def __init__(
        self,
        path_or_dataset,
        tokenizer=None,
        split=None,
        seq_length=1024,
        collate_fn=None,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        micro_batch_size=2,
        global_batch_size=2,
        pad_token_id=0,
        use_dist_sampler=False,
        train_aliases=["train", "training"],
        test_aliases=["test", "testing"],
        val_aliases=["val", "validation", "valid", "eval"],
        pad_seq_len_divisible=None,
        seed=1234,
        do_validation=True,
        do_test=True,
        num_replicas=None,
        rank=None,
        dataset_kwargs=None,
        **kwargs,
    ) -> None:
        assert pad_token_id is not None
        self.tokenizer = tokenizer
        self.seed = seed
        self.do_validation = do_validation
        self.do_test = do_test
        self.dataset_kwargs = dataset_kwargs or {}

        # A dataset usually will have several splits (e.g. train, val, test, etc).
        # We map synonym names to canonical names (train, test, val).
        # A synonym can be a prefix/suffixed word e.g. train <> training.
        split_aliases = {
            "train": train_aliases,
            "test": test_aliases,
            "val": val_aliases,
        }

        # self.dataset_splits will hold the actual dataset for each split.
        if isinstance(path_or_dataset, str):
            logging.info("Loading HF dataset from {}, this may take a moment.".format(path_or_dataset))
            dataset = load_dataset(path_or_dataset, split=split, **kwargs)
        elif isinstance(path_or_dataset, Dataset) or isinstance(path_or_dataset, DatasetDict):
            logging.info("Using passed HF dataset {}".format(path_or_dataset))
            dataset = path_or_dataset
        else:
            raise ValueError(
                "Expected `path_or_dataset` to be str, Dataset, DatasetDict, but got {}".format(type(path_or_dataset))
            )

        self.dataset_splits = make_dataset_splits(dataset, split, split_aliases)

        if collate_fn is None:
            self._collate_fn = lambda x: HFDatasetBuilder.collate_fn(
                x, pad_token_id=pad_token_id, pad_seq_len_divisible=pad_seq_len_divisible
            )
        else:
            self._collate_fn = collate_fn

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.global_batch_size = global_batch_size
        self.pad_token_id = pad_token_id
        self.use_dist_sampler = use_dist_sampler
        self.pad_seq_len_divisible = pad_seq_len_divisible

        # For distributed sampling
        self.num_replicas = num_replicas
        self.rank = rank

    @staticmethod
    def from_dict(dataset_dict, split, **kwargs):
        """Create a builder from a dictionary of data.

        Args:
            dataset_dict: Dictionary containing the dataset data.
            split: The split to use (e.g., "train").
            **kwargs: Additional arguments to pass to the constructor.

        Returns:
            HFDatasetBuilder: A new builder instance.
        """
        dataset = Dataset.from_dict(dataset_dict)
        return HFDatasetBuilder(path_or_dataset=dataset, split=split, **kwargs)

    @staticmethod
    def collate_fn(batch, pad_token_id=0, pad_seq_len_divisible=None):
        """Default batch collator that handles padding and batching.

        Args:
            batch: A batch of examples.
            pad_token_id: The token ID to use for padding.
            pad_seq_len_divisible: If provided, pad sequence length to be divisible by this value.

        Returns:
            dict: A dictionary containing batched tensors.
        """
        return {
            key: batchify(
                torch.LongTensor(
                    pad_within_micro(
                        extract_key_from_dicts(batch, key),
                        pad_token_id if key != 'loss_mask' else 0,
                        pad_seq_len_divisible,
                    )
                )
            )
            for key in batch[0].keys()
        }

    def prepare_data(self) -> None:
        """Prepare data if needed. This method is a placeholder for more complex preparation logic."""
        pass

    def build(self) -> list[Optional[Any]]:
        """Build train, validation, and test datasets.

        This method creates the necessary datasets based on the configuration.

        Returns:
            list[Optional[Any]]: A list containing the train, validation, and test datasets.
                                Any of these may be None if not available.
        """
        # Prepare data if needed
        self.prepare_data()

        # Build and return datasets
        train_ds = self._create_dataloader(self.dataset_splits["train"], self._collate_fn) if self.dataset_splits["train"] is not None else None
        valid_ds = self._create_dataloader(self.dataset_splits["val"], self._collate_fn) if self.do_validation and self.dataset_splits["val"] is not None else None
        test_ds = self._create_dataloader(self.dataset_splits["test"], self._collate_fn) if self.do_test and self.dataset_splits["test"] is not None else None

        return [train_ds, valid_ds, test_ds]

    def get_data_sampler(self, dataset):
        """Get the appropriate data sampler based on configuration.

        Args:
            dataset: The dataset to create a sampler for.

        Returns:
            Optional[DistributedSampler]: A sampler for distributed training, or None.
        """
        if self.use_dist_sampler or has_dist_env_init_or_rank_env_var():
            return DistributedSampler(dataset, num_replicas=self.num_replicas, rank=self.rank)
        else:
            return None

    def _create_dataloader(self, dataset, collate_fn=None):
        """Create a PyTorch DataLoader for the dataset.

        Args:
            dataset: The dataset to create a loader for.
            collate_fn: The collation function to use.

        Returns:
            DataLoader: A PyTorch DataLoader for the dataset.
        """
        assert dataset is not None

        if collate_fn is None:
            collate_fn = lambda x: HFDatasetBuilder.collate_fn(
                x, pad_token_id=self.pad_token_id, pad_seq_len_divisible=self.pad_seq_len_divisible
            )

        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
            batch_size=self.micro_batch_size,
            sampler=self.get_data_sampler(dataset),
        )

    def map(self, function=None, split_names=None, **kwargs):
        """Apply a function to all or selected splits of the dataset.

        Args:
            function: The function to apply to each example.
            split_names: The splits to apply the function to. If None, apply to all splits.
            **kwargs: Additional arguments to pass to the dataset's map method.
        """
        if isinstance(split_names, str):
            split_names = [split_names]
        elif isinstance(split_names, list):
            pass
        elif split_names is None:
            split_names = self.dataset_splits.keys()
        else:
            raise ValueError("split_names must None/str/list")

        for split_name in split_names:
            if not self.dataset_splits[split_name] is None:
                self.dataset_splits[split_name] = self.dataset_splits[split_name].map(function, **kwargs)

    @property
    def train(self):
        """Get the training dataset.

        Returns:
            The training dataset.
        """
        return self.dataset_splits["train"]

    @property
    def val(self):
        """Get the validation dataset.

        Returns:
            The validation dataset.
        """
        return self.dataset_splits["val"]

    @property
    def test(self):
        """Get the test dataset.

        Returns:
            The test dataset.
        """
        return self.dataset_splits["test"]