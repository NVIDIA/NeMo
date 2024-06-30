# coding=utf-8
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

"""Utilities for pipeline model parallel."""
"""Copied from https://github.com/NVIDIA/apex/blob/7b73b12361068a10b0f44844534613f252a5ea75/apex/transformer/pipeline_parallel/utils.py#L58 
   and https://github.com/NVIDIA/apex/blob/7b73b12361068a10b0f44844534613f252a5ea75/apex/transformer/microbatches.py
   and https://github.com/NVIDIA/apex/blob/7b73b12361068a10b0f44844534613f252a5ea75/apex/transformer/tensor_parallel/layers.py"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
from torch.nn.parallel import DistributedDataParallel

_GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
_GLOBAL_AUTORESUME = None


Shape = Union[List[int], torch.Size]

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel": False,
    "partition_dim": -1,
    "partition_stride": 1,
}


def get_num_microbatches():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor: torch.Tensor) -> None:
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def build_num_microbatches_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
):
    # Constant num micro-batches.
    if rampup_batch_size is None:
        num_microbatches_calculator = ConstantNumMicroBatches(global_batch_size, micro_batch_size, data_parallel_size)
        '''if rank == 0:
            _logger.info(
                "setting number of micro-batches to constant {}".format(
                    num_microbatches_calculator.get()
                )
            )'''

    else:
        assert len(rampup_batch_size) == 3, (
            "expected the following "
            "format: --rampup-batch-size <start batch size> "
            "<batch size incerement> <ramp-up samples>"
        )
        start_batch_size = int(rampup_batch_size[0])
        batch_size_increment = int(rampup_batch_size[1])
        ramup_samples = int(rampup_batch_size[2])
        if rank == 0:
            _logger.info(
                "will use batch size rampup starting from global batch "
                "size {} to global batch size {} with batch size increments "
                "{} over {} samples.".format(
                    start_batch_size,
                    global_batch_size,
                    batch_size_increment,
                    ramup_samples,
                )
            )
        num_microbatches_calculator = RampupBatchsizeNumMicroBatches(
            start_batch_size,
            batch_size_increment,
            ramup_samples,
            global_batch_size,
            micro_batch_size,
            data_parallel_size,
        )

    return num_microbatches_calculator


class NumMicroBatchesCalculator(ABC):
    def __init__(self):
        self.num_micro_batches = None
        self.current_global_batch_size = None

    def get(self):
        return self.num_micro_batches

    def get_current_global_batch_size(self):
        return self.current_global_batch_size

    @abstractmethod
    def update(self, consumed_samples, consistency_check):
        pass


class ConstantNumMicroBatches(NumMicroBatchesCalculator):
    def __init__(self, global_batch_size, micro_batch_size, data_parallel_size):
        micro_batch_times_data_parallel = micro_batch_size * data_parallel_size
        assert (
            global_batch_size % micro_batch_times_data_parallel == 0
        ), "global batch size ({}) is not divisible by micro batch size ({})" " times data parallel size ({})".format(
            global_batch_size, micro_batch_size, data_parallel_size
        )
        self.num_micro_batches = global_batch_size // micro_batch_times_data_parallel
        assert self.num_micro_batches >= 1
        self.current_global_batch_size = global_batch_size

        self.micro_batch_size = micro_batch_size

    def update(self, consumed_samples, consistency_check):
        pass


def listify_model(model: Union[torch.nn.Module, List[torch.nn.Module]]) -> List[torch.nn.Module]:
    if isinstance(model, list):
        return model
    return [model]


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is not None, "{} is not initialized.".format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    assert var is None, "{} is already initialized.".format(name)


def setup_microbatch_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
) -> None:
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR
    _ensure_var_is_not_initialized(_GLOBAL_NUM_MICROBATCHES_CALCULATOR, 'num microbatches calculator')

    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(
        rank, rampup_batch_size, global_batch_size, micro_batch_size, data_parallel_size
    )


def _reconfigure_microbatch_calculator(
    rank: int,
    rampup_batch_size: Optional[List[int]],
    global_batch_size: int,
    micro_batch_size: int,
    data_parallel_size: int,
) -> None:
    if torch.distributed.get_rank() == 0:
        import warnings

        warnings.warn("This function is only for unittest")
    global _GLOBAL_NUM_MICROBATCHES_CALCULATOR

    _GLOBAL_NUM_MICROBATCHES_CALCULATOR = build_num_microbatches_calculator(
        rank, rampup_batch_size, global_batch_size, micro_batch_size, data_parallel_size
    )


def get_micro_batch_size():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.micro_batch_size


def get_num_microbatches():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get()


def get_current_global_batch_size():
    return _GLOBAL_NUM_MICROBATCHES_CALCULATOR.get_current_global_batch_size()


def update_num_microbatches(consumed_samples, consistency_check=True):
    _GLOBAL_NUM_MICROBATCHES_CALCULATOR.update(consumed_samples, consistency_check)


# note (mkozuki): Comment out in favor of `get_kth_microbatch`
def _split_batch_into_microbatch(
    batch: List[torch.Tensor],
    *,
    _micro_batch_size: Optional[int] = None,
    _global_batch_size: Optional[int] = None,
) -> List[List[torch.Tensor]]:
    micro_batch_size = _micro_batch_size
    global_batch_size = _global_batch_size
    if micro_batch_size is None:
        micro_batch_size = get_micro_batch_size()
    if global_batch_size is None:
        global_batch_size = get_current_global_batch_size()
    for i in range(0, global_batch_size, micro_batch_size):
        yield [x[i * micro_batch_size : (i + 1) * micro_batch_size] for x in batch]


# TODO(mkozuki): Support non-tensor local minibatches?
def get_kth_microbatch(batch: Optional[List[torch.Tensor]], k: int) -> List[torch.Tensor]:
    """Create a list of microbatches from a list of local minibatches.

    This function creates a list of `k`th microbatches from a list of local minibatches.
    `a local minibatch` consists of `global_batch_size / data_parallel_size` samples.
    """
    if batch is None or not isinstance(batch, (List, Tuple)):
        return batch
    micro_batch_size = get_micro_batch_size()
    start = k * micro_batch_size
    end = start + micro_batch_size
    microbatch = list()
    for x in batch:
        size = x.size(0)
        assert size > start and size >= end
        microbatch.append(x[start:end])
    assert len(microbatch) > 0
    return microbatch


def get_autoresume():
    return _GLOBAL_AUTORESUME


def print_rank_0(message: str) -> None:
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def is_last_rank():
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)


def param_is_not_shared(param: torch.nn.Parameter) -> bool:
    return getattr(param, "shared", False)


def unwrap_model(model, module_instances=(DistributedDataParallel,)):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


# NOTE (mkozuki): APEX doesn't have anything equivalent for
# `_GLOBAL_ADLR_AUTORESUME` like Megatron-LM.
# def check_adlr_autoresume_termination(iteration, model, optimizer, lr_scheduler, save: bool):
#     """Check for autoresume signal and exit if it is received."""
#     from apex.ppu.checkpointing import save_checkpoint
#
#     autoresume = get_adlr_autoresume()
#     # Add barrier to ensure consistency.
#     torch.distributed.barrier()
#     if autoresume.termination_requested():
#         if save:
#             save_checkpoint(iteration, model, optimizer, lr_scheduler)
#         print_rank_0(">>> autoresume termination request found!")
#         if torch.distributed.get_rank() == 0:
#             autoresume.request_resume()
#         print_rank_0(">>> training terminated. Returning")
#         sys.exit(0)


def get_ltor_masks_and_position_ids(data, eod_token, reset_position_ids, reset_attention_mask, eod_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)).view(
        att_mask_batch, 1, seq_length, seq_length
    )

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1) :] -= i + 1 - prev_index
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids
