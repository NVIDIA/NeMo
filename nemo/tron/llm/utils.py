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

from typing import Dict, Iterable

import torch
from megatron.core import parallel_state

from nemo.tron.config import ConfigContainer, FinetuningDatasetConfig


def get_batch_from_iterator(data_iterator: Iterable) -> Dict[str, torch.Tensor]:
    assert data_iterator is not None, "data_iterator must not be None"

    data = next(data_iterator)

    batch = {
        "tokens": data["tokens"].cuda(non_blocking=True),
        "labels": data["labels"].cuda(non_blocking=True),
        "loss_mask": data["loss_mask"].cuda(non_blocking=True),
        "attention_mask": None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking=True),
        "position_ids": data["position_ids"].cuda(non_blocking=True),
    }

    return batch


def get_batch_on_this_tp_rank(data_iterator: Iterable, cfg: ConfigContainer) -> Dict[str, torch.Tensor]:
    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                parallel_state.get_tensor_model_parallel_src_rank(),
                group=parallel_state.get_tensor_model_parallel_group(),
            )

    if parallel_state.get_tensor_model_parallel_rank() == 0:
        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None

        batch = {
            "tokens": data["tokens"].cuda(non_blocking=True),
            "labels": data["labels"].cuda(non_blocking=True),
            "loss_mask": data["loss_mask"].cuda(non_blocking=True),
            "attention_mask": None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking=True),
            "position_ids": data["position_ids"].cuda(non_blocking=True),
        }

        if cfg.model_config.pipeline_model_parallel_size == 1:
            _broadcast(batch["tokens"])
            _broadcast(batch["labels"])
            _broadcast(batch["loss_mask"])
            _broadcast(batch["attention_mask"])
            _broadcast(batch["position_ids"])

        elif parallel_state.is_pipeline_first_stage():
            _broadcast(batch["tokens"])
            _broadcast(batch["attention_mask"])
            _broadcast(batch["position_ids"])

        elif parallel_state.is_pipeline_last_stage():
            _broadcast(batch["labels"])
            _broadcast(batch["loss_mask"])
            _broadcast(batch["attention_mask"])

    else:
        mbs = cfg.train_config.micro_batch_size
        seq_length = cfg.model_config.seq_length
        tokens = torch.empty(
            (mbs, seq_length),
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        labels = torch.empty(
            (mbs, seq_length),
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        loss_mask = torch.empty(
            (mbs, seq_length),
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )
        if isinstance(cfg.dataset_config, FinetuningDatasetConfig) or cfg.dataset_config.create_attention_mask:
            attention_mask = torch.empty(
                (
                    mbs,
                    1,
                    seq_length,
                    seq_length,
                ),
                dtype=torch.bool,
                device=torch.cuda.current_device(),
            )
        else:
            attention_mask = None
        position_ids = torch.empty(
            (mbs, seq_length),
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )

        if cfg.model_config.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif parallel_state.is_pipeline_first_stage():
            labels = None
            loss_mask = None

            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif parallel_state.is_pipeline_last_stage():
            tokens = None
            position_ids = None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)

        batch = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    return batch
