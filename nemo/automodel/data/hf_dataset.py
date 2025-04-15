import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from nemo.utils import logging


def cyclic_iter(iter):
    """Create a cyclic iterator that loops infinitely over the base iterator.

    Args:
        iter: The base iterator.

    Yields:
        The next element from the base iterator, looping back to the start when exhausted.
    """
    while True:
        for x in iter:
            yield x


def clean_split(name):
    """Removes split from name

    Args:
        name (str): partition name (e.g. "train[:100]")

    Returns:
        str: return partition name without any selector (e.g. "train").
    """
    if "[" in name:
        name = name.split("[")[0]
    if "+" in name:
        name = name.split("+")[0]
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
    return dist.is_initialized() or int(os.environ.get("WORLD_SIZE", "0")) > 1


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


@dataclass(kw_only=True)
class DataloaderConfig:
    """Base configuration for dataloaders."""

    dataloader_type: Optional[str] = None
    """Single pass vs multiple pass data loader"""

    num_workers: int = 8
    """Dataloader number of workers."""

    data_sharding: bool = True
    """Disable data sharding."""

    pin_memory: bool = True
    """Whether to pin memory during data loading for faster GPU training."""

    persistent_workers: bool = False
    """Whether to keep data loading workers persistent across epochs."""


@dataclass(kw_only=True)
class HFDatasetBuilder(DataloaderConfig):
    """A builder class for loading and managing datasets from the Hugging Face datasets library.

    This class provides a builder-like interface for creating datasets and dataloaders
    from Hugging Face datasets. It handles dataset loading, splitting, and creation of
    PyTorch DataLoaders with appropriate collation functions and samplers.
    """

    path_or_dataset: Union[str, Dataset, DatasetDict]
    """The dataset name or a preloaded dataset."""

    split: Optional[Union[str, list[str]]] = None
    """The dataset split(s) to load (e.g., "train" or ["train", "validation"])."""

    seq_length: int = 1024
    """Maximum sequence length for tokenized inputs."""

    collate_fn: Optional[Callable] = None
    """Custom function for batching data."""

    pad_token_id: int = 0
    """Token ID used for padding sequences."""

    use_dist_sampler: bool = False
    """Whether to enable distributed sampling."""

    train_aliases: list[str] = field(default_factory=lambda: ["train", "training"])
    """Alternative names for the training split."""

    test_aliases: list[str] = field(default_factory=lambda: ["test", "testing"])
    """Alternative names for the test split."""

    val_aliases: list[str] = field(default_factory=lambda: ["val", "validation", "valid", "eval"])
    """Alternative names for the validation split."""

    pad_seq_len_divisible: Optional[int] = None
    """If provided, pad sequence length to be divisible by this value."""

    seed: int = 1234
    """Random seed for shuffling and other random operations."""

    do_validation: bool = True
    """Whether to prepare validation data."""

    do_test: bool = True
    """Whether to prepare test data."""

    dataset_kwargs: Optional[dict[str, Any]] = None
    """Additional arguments for dataset creation."""

    def __post_init__(self) -> None:
        """Initialize the builder after dataclass initialization."""
        assert self.pad_token_id is not None
        self.dataset_kwargs = self.dataset_kwargs or {}

        # A dataset usually will have several splits (e.g. train, val, test, etc).
        # We map synonym names to canonical names (train, test, val).
        # A synonym can be a prefix/suffixed word e.g. train <> training.
        split_aliases = {
            "train": self.train_aliases,
            "test": self.test_aliases,
            "val": self.val_aliases,
        }

        # self.dataset_splits will hold the actual dataset for each split.
        if isinstance(self.path_or_dataset, str):
            logging.info("Loading HF dataset from {}, this may take a moment.".format(self.path_or_dataset))
            dataset = load_dataset(self.path_or_dataset, split=self.split, **(self.dataset_kwargs or {}))
        elif isinstance(self.path_or_dataset, Dataset) or isinstance(self.path_or_dataset, DatasetDict):
            logging.info("Using passed HF dataset {}".format(self.path_or_dataset))
            dataset = self.path_or_dataset
        else:
            raise ValueError(
                "Expected `path_or_dataset` to be str, Dataset, DatasetDict, but got {}".format(
                    type(self.path_or_dataset)
                )
            )

        self.dataset_splits = make_dataset_splits(dataset, self.split, split_aliases)

        if self.collate_fn is None:
            self._collate_fn = lambda x: HFDatasetBuilder.default_collate_fn(
                x, pad_token_id=self.pad_token_id, pad_seq_len_divisible=self.pad_seq_len_divisible
            )
        else:
            self._collate_fn = self.collate_fn

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
    def default_collate_fn(batch, pad_token_id=0, pad_seq_len_divisible=None):
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
                        pad_token_id if key != "loss_mask" else 0,
                        pad_seq_len_divisible,
                    )
                )
            )
            for key in batch[0].keys()
        }

    def prepare_data(self) -> None:
        """Prepare data if needed. This method is a placeholder for more complex preparation logic."""
        pass

    def build(
        self,
        rank: int,
        world_size: int,
        micro_batch_size: int,
    ) -> list[Optional[Any]]:
        """Build train, validation, and test datasets.

        This method creates the necessary datasets based on the configuration.

        Returns:
            list[Optional[Any]]: A list containing the train, validation, and test datasets.
                                Any of these may be None if not available.
        """
        # Prepare data if needed
        self.prepare_data()

        # Build and return datasets
        train_ds = (
            self._create_dataloader(self.dataset_splits["train"], micro_batch_size, rank, world_size, self._collate_fn)
            if self.dataset_splits["train"] is not None
            else None
        )
        valid_ds = (
            self._create_dataloader(self.dataset_splits["val"], micro_batch_size, rank, world_size, self._collate_fn)
            if self.do_validation and self.dataset_splits["val"] is not None
            else None
        )
        test_ds = (
            self._create_dataloader(self.dataset_splits["test"], micro_batch_size, rank, world_size, self._collate_fn)
            if self.do_test and self.dataset_splits["test"] is not None
            else None
        )

        return [train_ds, valid_ds, test_ds]

    def get_data_sampler(self, dataset, rank: int, world_size: int):
        """Get the appropriate data sampler based on configuration.

        Args:
            dataset: The dataset to create a sampler for.

        Returns:
            Optional[DistributedSampler]: A sampler for distributed training, or None.
        """
        if self.use_dist_sampler or has_dist_env_init_or_rank_env_var():
            return DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        else:
            return None

    def _create_dataloader(self, dataset, micro_batch_size: int, rank: int, world_size: int, collate_fn=None):
        """Create a PyTorch DataLoader for the dataset.

        Args:
            dataset: The dataset to create a loader for.
            collate_fn: The collation function to use.

        Returns:
            DataLoader: A PyTorch DataLoader for the dataset.
        """
        assert dataset is not None

        if collate_fn is None:
            collate_fn = lambda x: HFDatasetBuilder.default_collate_fn(
                x, pad_token_id=self.pad_token_id, pad_seq_len_divisible=self.pad_seq_len_divisible
            )

        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
            batch_size=micro_batch_size,
            sampler=self.get_data_sampler(dataset, rank, world_size),
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
            if self.dataset_splits[split_name] is not None:
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

    def build_iterators(
        self,
        rank: int,
        world_size: int,
        micro_batch_size: int,
    ):
        """Build training, validation, and test data iterators from a configuration object.

        This function creates iterators for training, validation, and testing data loaders
        based on the provided configuration.

        Args:
            config: The configuration object containing all parameters.

        Returns:
            tuple: A tuple containing (train_data_iterator, valid_data_iterator, test_data_iterator)
        """
        # Extract dataset configuration

        # Build dataloaders
        train_dataloader, valid_dataloader, test_dataloader = self.build(rank, world_size, micro_batch_size)

        # Determine dataloader type from config
        dl_type = self.dataloader_type
        assert dl_type in ["single", "cyclic", "external"]

        # Build iterators
        if train_dataloader is not None:
            if dl_type == "single":
                train_data_iterator = iter(train_dataloader)
            elif dl_type == "cyclic":
                train_data_iterator = iter(cyclic_iter(train_dataloader))
            elif dl_type == "external":
                # External dataloader is passed through
                train_data_iterator = train_dataloader
            else:
                raise RuntimeError("unexpected dataloader type")
        else:
            train_data_iterator = None

        # For validation, always use cyclic iterator
        if valid_dataloader is not None:
            valid_data_iterator = iter(cyclic_iter(valid_dataloader))
        else:
            valid_data_iterator = None

        # For test, use the specified dataloader type
        if test_dataloader is not None:
            if dl_type == "single":
                test_data_iterator = iter(test_dataloader)
            elif dl_type == "cyclic":
                test_data_iterator = iter(cyclic_iter(test_dataloader))
            elif dl_type == "external":
                test_data_iterator = test_dataloader
            else:
                raise RuntimeError("unexpected dataloader type")
        else:
            test_data_iterator = None

        return train_data_iterator, valid_data_iterator, test_data_iterator
