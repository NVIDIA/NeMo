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

import contextlib
from functools import partial
from typing import Iterable, Optional

import torch
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import get_batch_on_this_cp_rank

from nemo.tron.config import ConfigContainer
from nemo.tron.llm.common import get_batch_on_this_tp_rank
from nemo.tron.state import GlobalState

SPIKY_LOSS_FACTOR = 10


def get_batch(data_iterator, cfg: ConfigContainer):
    """Generate a batch."""

    if (not parallel_state.is_pipeline_first_stage()) and (not parallel_state.is_pipeline_last_stage()):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator, cfg)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, cfg: ConfigContainer):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    total_tokens = loss_mask.sum()
    loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

    if cfg.model_config.context_parallel_size > 1:
        torch.distributed.all_reduce(loss, group=parallel_state.get_context_parallel_group())

    rerun_state_machine = get_rerun_state_machine()
    if cfg.megatron_lm_config.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if cfg.megatron_lm_config.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss[0],
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    torch.distributed.all_reduce(reporting_loss, group=parallel_state.get_data_parallel_group())

    local_num_tokens = loss[1].clone().detach().to(torch.int)
    return (
        loss[0] * cfg.model_config.context_parallel_size,
        local_num_tokens,
        {"lm loss": (reporting_loss[0], reporting_loss[1])},
    )


def forward_step(
    data_iterator: Iterable,
    model: GPTModel,
    cfg: ConfigContainer,
    state: Optional[GlobalState] = None,
):
    """Forward training step.

    Args:
        data_iterator (Iterable): Input data iterator
        model (GPTModel): The GPT Model
        cfg (ConfigContainer): Full config
        state (GlobalState): State of this run
    """

    timers = None
    straggler_timer = None
    if state:
        timers = state.timers
        straggler_timer = state.straggler_timer

    if timers:
        timers("batch-generator", log_level=2).start()
    if straggler_timer:
        with straggler_timer(bdata=True):
            tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator, cfg)
    else:
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator, cfg)
    if timers:
        timers("batch-generator").stop()

    if straggler_timer:
        with straggler_timer:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    else:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask=loss_mask, cfg=cfg)
