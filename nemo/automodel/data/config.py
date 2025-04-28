from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union


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
class HFDatasetConfig(DataloaderConfig):
    """Configuration for Hugging Face datasets.

    This configuration class defines the parameters needed to load and process
    datasets from the Hugging Face datasets library.
    """

    path_or_dataset: Optional[Union[str, Any]] = None
    """Path to the dataset or the dataset object."""

    split: Optional[Union[str, list[str]]] = None
    """The split to use (e.g., "train" or ["train", "validation"])."""

    seq_length: int = 1024
    """Maximum sequence length for tokenized inputs."""

    seed: int = 1234
    """Random seed for shuffling and other random operations."""

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

    do_validation: bool = True
    """Whether to prepare validation data."""

    do_test: bool = True
    """Whether to prepare test data."""

    dataset_kwargs: Optional[dict[str, Any]] = None
    """Additional arguments for dataset creation."""

    additional_kwargs: Optional[dict[str, Any]] = field(default_factory=dict)
    """Additional arguments to pass to the dataset loader."""
