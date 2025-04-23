from typing import Callable

from nemo.automodel.data.config import HFDatasetConfig
from nemo.tron.utils.common_utils import print_rank_0


def hf_train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int], dataset_config: HFDatasetConfig, tokenizer
):
    """Build the train test and validation datasets from a Hugging Face dataset.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
        dataset_config: Configuration for the dataset.
        tokenizer: The tokenizer to use for processing text.
    """
    print_rank_0(
        f"> building train, validation, and test datasets for Hugging Face dataset from {dataset_config.path_or_dataset} ..."
    )

    from nemo.automodel.data.hf_dataset import HFDatasetBuilder

    # Create dataset builder
    builder = HFDatasetBuilder(
        path_or_dataset=dataset_config.path_or_dataset,
        tokenizer=tokenizer,
        split=dataset_config.split,
        seq_length=dataset_config.seq_length,
        num_workers=dataset_config.num_workers,
        pin_memory=dataset_config.pin_memory,
        persistent_workers=dataset_config.persistent_workers,
        pad_token_id=dataset_config.pad_token_id,
        use_dist_sampler=dataset_config.use_dist_sampler,
        train_aliases=dataset_config.train_aliases,
        test_aliases=dataset_config.test_aliases,
        val_aliases=dataset_config.val_aliases,
        pad_seq_len_divisible=dataset_config.pad_seq_len_divisible,
        seed=dataset_config.seed,
        do_validation=dataset_config.do_validation,
        do_test=dataset_config.do_test,
        dataset_kwargs=dataset_config.dataset_kwargs,
        **(dataset_config.additional_kwargs or {}),
    )

    # Build datasets
    train_ds, valid_ds, test_ds = builder.build()

    print_rank_0(f"> finished creating Hugging Face dataset from {dataset_config.path_or_dataset} ...")

    return train_ds, valid_ds, test_ds


# Registry mapping dataset config types to their provider functions
REGISTRY = {
    HFDatasetConfig: hf_train_valid_test_datasets_provider,
}


def get_dataset_provider(
    dataset_config,
) -> Callable:
    """Get the appropriate dataset provider function for the given dataset config.

    Args:
        dataset_config: The dataset configuration object.

    Returns:
        A callable function that can build train, validation, and test datasets.

    Raises:
        ValueError: If no provider is registered for the given dataset config type.
    """
    config_type = type(dataset_config)
    if config_type not in REGISTRY:
        raise ValueError(f"No dataset provider registered for config type {config_type.__name__}")

    return REGISTRY[config_type]