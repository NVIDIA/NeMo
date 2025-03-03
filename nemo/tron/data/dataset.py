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

import json
from typing import Optional

import torch
from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.gpt_dataset import GPTDataset, MockGPTDataset
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.rerun_state_machine import RerunDataIterator

from nemo.tron.config import ConfigContainer
from nemo.tron.data.samplers import build_pretraining_data_loader
from nemo.tron.state import TrainState
from nemo.tron.utils.common_utils import print_rank_0
from nemo.tron.utils.sig_utils import DistributedSignalHandler


def get_blend_and_blend_per_split(
    data_paths: Optional[list[str]] = None,
    data_args_path: Optional[str] = None,
    per_split_data_args_path: Optional[str] = None,
    train_data_paths: Optional[list[str]] = None,
    valid_data_paths: Optional[list[str]] = None,
    test_data_paths: Optional[list[str]] = None,
):
    """Get blend and blend_per_split from passed-in arguments."""
    use_data_path = data_paths is not None or data_args_path is not None
    use_per_split_data_path = (
        any(elt is not None for elt in [train_data_paths, valid_data_paths, test_data_paths])
        or per_split_data_args_path is not None
    )

    blend = None
    blend_per_split = None
    if use_data_path:
        if data_args_path is not None:
            assert data_paths is None
            with open(data_args_path, "r") as f:
                blend = get_blend_from_list(f.read().split())
        else:
            assert data_paths is not None
            blend = get_blend_from_list(data_paths)
    elif use_per_split_data_path:
        if per_split_data_args_path is not None:
            with open(per_split_data_args_path, "r") as f:
                per_split_data_args = json.load(f)
                # Each element in blend_per_split should be a list of files (and optional
                # weights), so split string if needed.
                for split in ["train", "valid", "test"]:
                    if isinstance(per_split_data_args[split], str):
                        per_split_data_args[split] = per_split_data_args[split].split()

                blend_per_split = [
                    get_blend_from_list(per_split_data_args["train"]),
                    get_blend_from_list(per_split_data_args["valid"]),
                    get_blend_from_list(per_split_data_args["test"]),
                ]
        else:
            blend_per_split = [
                get_blend_from_list(train_data_paths),
                get_blend_from_list(valid_data_paths),
                get_blend_from_list(test_data_paths),
            ]
    else:
        blend, blend_per_split = None, None

    return blend, blend_per_split


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def train_valid_test_datasets_provider(
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


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


def get_train_valid_test_num_samples(cfg: ConfigContainer):
    """Train/valid/test num samples."""

    # Number of train/valid/test samples.
    train_samples = cfg.train_config.train_iters * cfg.train_config.global_batch_size
    eval_iters = (cfg.train_config.train_iters // cfg.train_config.eval_interval + 1) * cfg.train_config.eval_iters
    test_iters = cfg.train_config.eval_iters

    return (
        train_samples,
        eval_iters * cfg.train_config.global_batch_size,
        test_iters * cfg.train_config.global_batch_size,
    )


def build_train_valid_test_datasets(cfg: ConfigContainer, build_train_valid_test_datasets_provider):
    """Build pretraining datasets."""
    train_valid_test_num_samples = get_train_valid_test_num_samples(cfg)
    print_rank_0(" > datasets target sizes (minimum size):")
    print_rank_0("    train:      {}".format(train_valid_test_num_samples[0]))
    print_rank_0("    validation: {}".format(train_valid_test_num_samples[1]))
    print_rank_0("    test:       {}".format(train_valid_test_num_samples[2]))
    return build_train_valid_test_datasets_provider(train_valid_test_num_samples, cfg.dataset_config)


def build_train_valid_test_data_loaders(
    cfg: ConfigContainer, train_state: TrainState, build_train_valid_test_datasets_provider
):
    """Build pretraining data loaders."""
    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0("> building train, validation, and test datasets ...")

    # Construct the data pipeline
    # Build datasets.
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        cfg=cfg, build_train_valid_test_datasets_provider=build_train_valid_test_datasets_provider
    )

    def worker_init_fn(_):
        DistributedSignalHandler().__enter__()

    maybe_worker_init_fn = worker_init_fn if cfg.train_config.exit_signal_handler_for_dataloader else None

    # Build dataloders.
    train_dataloader = build_pretraining_data_loader(
        train_ds,
        train_state.consumed_train_samples,
        cfg.dataset_config.dataloader_type,
        cfg.train_config.micro_batch_size,
        cfg.dataset_config.num_workers,
        cfg.dataset_config.data_sharding,
        worker_init_fn=maybe_worker_init_fn,
    )
    if cfg.train_config.skip_train:
        valid_dataloader = build_pretraining_data_loader(
            valid_ds,
            0,
            cfg.dataset_config.dataloader_type,
            cfg.train_config.micro_batch_size,
            cfg.dataset_config.num_workers,
            cfg.dataset_config.data_sharding,
            worker_init_fn=maybe_worker_init_fn,
        )
    else:
        valid_dataloader = build_pretraining_data_loader(
            valid_ds,
            train_state.consumed_valid_samples,
            "cyclic",
            cfg.train_config.micro_batch_size,
            cfg.dataset_config.num_workers,
            cfg.dataset_config.data_sharding,
            worker_init_fn=maybe_worker_init_fn,
        )
    test_dataloader = build_pretraining_data_loader(
        test_ds,
        0,
        cfg.dataset_config.dataloader_type,
        cfg.train_config.micro_batch_size,
        cfg.dataset_config.num_workers,
        cfg.dataset_config.data_sharding,
    )

    # Flags to know if we need to do training/validation/testing.
    do_train = train_dataloader is not None and cfg.train_config.train_iters > 0
    do_valid = valid_dataloader is not None and cfg.train_config.eval_iters > 0
    do_test = test_dataloader is not None and cfg.train_config.eval_iters > 0
    flags = torch.tensor([int(do_train), int(do_valid), int(do_test)], dtype=torch.long, device="cuda")

    torch.distributed.broadcast(flags, 0)

    train_state.do_train = train_state.do_train or flags[0].item()
    train_state.do_valid = train_state.do_valid or flags[1].item()
    train_state.do_test = train_state.do_test or flags[2].item()

    return train_dataloader, valid_dataloader, test_dataloader


def build_train_valid_test_data_iterators(
    cfg: ConfigContainer, train_state: TrainState, build_train_valid_test_datasets_provider
):
    """Build pretraining data iterators."""

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(
        cfg=cfg,
        train_state=train_state,
        build_train_valid_test_datasets_provider=build_train_valid_test_datasets_provider,
    )

    # Build iterators.
    dl_type = cfg.dataset_config.dataloader_type
    assert dl_type in ["single", "cyclic", "external"]

    def _get_iterator(dataloader_type, dataloader):
        """Return dataset iterator."""
        if dataloader_type == "single":
            return RerunDataIterator(iter(dataloader))
        elif dataloader_type == "cyclic":
            return RerunDataIterator(iter(cyclic_iter(dataloader)))
        elif dataloader_type == "external":
            # External dataloader is passed through. User is expected to define how to iterate.
            if isinstance(dataloader, list):
                return [RerunDataIterator(d) for d in dataloader]
            else:
                return RerunDataIterator(dataloader)
        else:
            raise RuntimeError("unexpected dataloader type")

    if train_dataloader is not None:
        train_data_iterator = _get_iterator(dl_type, train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = _get_iterator("cyclic", valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = _get_iterator(dl_type, test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator


def setup_data_iterators(
    cfg: ConfigContainer, train_state: TrainState, model_length: int, train_valid_test_dataset_provider
):
    """Setup data iterators."""
    if cfg.model_config.virtual_pipeline_model_parallel_size is not None:
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for i in range(model_length):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            iterators = build_train_valid_test_data_iterators(
                cfg=cfg,
                train_state=train_state,
                build_train_valid_test_datasets_provider=train_valid_test_dataset_provider,
            )
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator = build_train_valid_test_data_iterators(
            cfg=cfg,
            train_state=train_state,
            build_train_valid_test_datasets_provider=train_valid_test_dataset_provider,
        )

    return train_data_iterator, valid_data_iterator, test_data_iterator
