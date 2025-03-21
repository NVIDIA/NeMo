from typing import Callable

from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.gpt_dataset import GPTDataset, MockGPTDataset

from nemo.tron.config import FinetuningDatasetConfig, GPTDatasetConfig
from nemo.tron.data.finetuning_dataset import FinetuningDatasetBuilder
from nemo.tron.data.hf_dataset import HFDatasetBuilder, HFDatasetConfig
from nemo.tron.tokenizers.tokenizer import MegatronTokenizer
from nemo.tron.utils.common_utils import print_rank_0


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def pretrain_train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int], dataset_config: BlendedMegatronDatasetConfig
):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """

    if dataset_config.mock:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, dataset_config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def hf_train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int], dataset_config: HFDatasetConfig, tokenizer: MegatronTokenizer
):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    print_rank_0(
        f"> building train, validation, and test datasets for Huggingface dataset {dataset_config.dataset_name} ..."
    )

    train_ds, valid_ds, test_ds = HFDatasetBuilder(
        dataset_name=dataset_config.dataset_name,
        tokenizer=tokenizer,
        is_built_on_rank=is_dataset_built_on_rank,
        process_example_fn=dataset_config.process_example_fn,
        dataset_subset=dataset_config.dataset_subset,
        dataset_root=dataset_config.dataset_root,
        split=dataset_config.split,
        seq_length=dataset_config.seq_length,
        max_train_samples=dataset_config.max_train_samples,
        packed_sequence_specs=dataset_config.packed_sequence_specs,
        force_redownload=dataset_config.force_redownload,
        val_proportion=dataset_config.val_proportion,
        split_val_from_train=dataset_config.split_val_from_train,
        delete_raw=dataset_config.delete_raw,
        dataset_kwargs=dataset_config.dataset_kwargs,
        hf_kwargs=dataset_config.hf_kwargs,
    ).build()

    print_rank_0(f"> finished creating Huggingface dataset {dataset_config.dataset_name} ...")

    return train_ds, valid_ds, test_ds


def finetuning_train_valid_test_datasets_provider(
    train_val_test_num_samples: list[int], dataset_config: FinetuningDatasetConfig, tokenizer: MegatronTokenizer
):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    print_rank_0(
        f"> building train, validation, and test datasets for Finetuning dataset from {dataset_config.dataset_root} ..."
    )

    train_ds, valid_ds, test_ds = FinetuningDatasetBuilder(
        dataset_root=dataset_config.dataset_root,
        tokenizer=tokenizer,
        is_built_on_rank=is_dataset_built_on_rank,
        seq_length=dataset_config.seq_length,
        max_train_samples=dataset_config.max_train_samples,
        packed_sequence_specs=dataset_config.packed_sequence_specs,
        dataset_kwargs=dataset_config.dataset_kwargs,
    ).build()

    print_rank_0(f"> finished creating Finetuning dataset from {dataset_config.dataset_root} ...")

    return train_ds, valid_ds, test_ds


REGISTRY = {
    GPTDatasetConfig: pretrain_train_valid_test_datasets_provider,
    HFDatasetConfig: hf_train_valid_test_datasets_provider,
    FinetuningDatasetConfig: finetuning_train_valid_test_datasets_provider,
}


def get_dataset_provider(
    dataset_config: FinetuningDatasetConfig | BlendedMegatronDatasetConfig | HFDatasetConfig,
) -> Callable:
    return REGISTRY[type(dataset_config)]
