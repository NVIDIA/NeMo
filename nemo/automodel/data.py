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

from typing import Callable

import torch

from nemo.automodel.config import ConfigContainer
from nemo.tron.data.loaders import build_train_valid_test_datasets, cyclic_iter
from nemo.tron.data.samplers import build_pretraining_data_loader
from nemo.tron.state import TrainState
from nemo.tron.utils.common_utils import print_rank_0
from nemo.tron.utils.sig_utils import DistributedSignalHandler


def build_train_valid_test_data_loaders(
    cfg: ConfigContainer,
    train_state: TrainState,
    data_parallel_rank: int,
    data_parallel_size: int,
    build_train_valid_test_datasets_provider: Callable,
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
        collate_fn=train_ds.collate_fn if hasattr(train_ds, "collate_fn") else None,
        pin_memory=cfg.dataset_config.pin_memory,
        persistent_workers=cfg.dataset_config.persistent_workers,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=data_parallel_size,
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
            collate_fn=valid_ds.collate_fn if hasattr(valid_ds, "collate_fn") else None,
            pin_memory=cfg.dataset_config.pin_memory,
            persistent_workers=cfg.dataset_config.persistent_workers,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
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
            collate_fn=valid_ds.collate_fn if hasattr(valid_ds, "collate_fn") else None,
            pin_memory=cfg.dataset_config.pin_memory,
            persistent_workers=cfg.dataset_config.persistent_workers,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
        )
    test_dataloader = build_pretraining_data_loader(
        test_ds,
        0,
        cfg.dataset_config.dataloader_type,
        cfg.train_config.micro_batch_size,
        cfg.dataset_config.num_workers,
        cfg.dataset_config.data_sharding,
        worker_init_fn=maybe_worker_init_fn,
        collate_fn=test_ds.collate_fn if hasattr(test_ds, "collate_fn") else None,
        pin_memory=cfg.dataset_config.pin_memory,
        persistent_workers=cfg.dataset_config.persistent_workers,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=data_parallel_size,
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
    cfg: ConfigContainer,
    train_state: TrainState,
    data_parallel_rank: int,
    data_parallel_size: int,
    build_train_valid_test_datasets_provider: Callable,
):
    """Build pretraining data iterators."""

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = build_train_valid_test_data_loaders(
        cfg=cfg,
        train_state=train_state,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=data_parallel_size,
        build_train_valid_test_datasets_provider=build_train_valid_test_datasets_provider,
    )

    # Build iterators.
    dl_type = cfg.dataset_config.dataloader_type
    assert dl_type in ["single", "cyclic", "external"]

    def _get_iterator(dataloader_type, dataloader):
        """Return dataset iterator."""
        if dataloader_type == "single":
            return iter(dataloader)
        elif dataloader_type == "cyclic":
            return iter(cyclic_iter(dataloader))
        elif dataloader_type == "external":
            # External dataloader is passed through. User is expected to define how to iterate.
            if isinstance(dataloader, list):
                return [iter(d) for d in dataloader]
            else:
                return dataloader
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
