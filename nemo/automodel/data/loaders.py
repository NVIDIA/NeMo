import torch

from nemo.automodel.data.config import HFDatasetConfig
from nemo.automodel.data.hf_dataset import HFDatasetBuilder
from nemo.tron.state import TrainState
from nemo.utils import logging


def build_train_valid_test_data_loaders(
    dataset_config: HFDatasetConfig,
    train_state: TrainState,
    rank: int,
    world_size: int,
    micro_batch_size: int,
    global_batch_size: int,
    train_iters: int,
    valid_iters: int,
    test_iters: int,
):
    """Build training, validation, and test data loaders from a configuration object.

    This function creates data loaders for training, validation, and testing
    using the HFDatasetBuilder class with parameters from the configuration.

    Args:
        config: The configuration object containing all parameters.

    Returns:
        tuple: A tuple containing (train_dataloader, valid_dataloader, test_dataloader)
    """
    logging.info("> Building train, validation, and test datasets...")

    # Extract dataset configuration

    # Initialize the dataset builder
    dataset_builder = HFDatasetBuilder(
        path_or_dataset=dataset_config.path_or_dataset,
        tokenizer=dataset_config.tokenizer,
        split=dataset_config.split,
        seq_length=dataset_config.seq_length,
        num_workers=dataset_config.num_workers,
        pin_memory=dataset_config.pin_memory,
        persistent_workers=dataset_config.persistent_workers,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
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
        rank=rank,
        num_replicas=world_size,
        **(dataset_config.additional_kwargs or {}),
    )

    # Build datasets and dataloaders
    train_dataloader, valid_dataloader, test_dataloader = dataset_builder.build()

    # Flags to know if we need to do training/validation/testing
    do_train = train_dataloader is not None and train_iters > 0
    do_valid = valid_dataloader is not None and valid_iters > 0
    do_test = test_dataloader is not None and test_iters > 0
    flags = torch.tensor([int(do_train), int(do_valid), int(do_test)], dtype=torch.long, device="cuda")

    torch.distributed.broadcast(flags, 0)

    train_state.do_train = train_state.do_train or flags[0].item()
    train_state.do_valid = train_state.do_valid or flags[1].item()
    train_state.do_test = train_state.do_test or flags[2].item()

    # Log info about dataset configuration
    logging.info(f"Training dataset ready: {do_train}")
    logging.info(f"Validation dataset ready: {do_valid}")
    logging.info(f"Test dataset ready: {do_test}")

    return train_dataloader, valid_dataloader, test_dataloader


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


def build_train_valid_test_data_iterators(
    dataset_config: HFDatasetConfig,
    train_state: TrainState,
    rank: int,
    world_size: int,
    micro_batch_size: int,
    global_batch_size: int,
    train_iters: int,
    valid_iters: int,
    test_iters: int,
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
    train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(
        dataset_config,
        train_state,
        rank,
        world_size,
        micro_batch_size,
        global_batch_size,
        train_iters,
        valid_iters,
        test_iters,
    )

    # Determine dataloader type from config
    dl_type = getattr(dataset_config, "dataloader_type", "single")
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
