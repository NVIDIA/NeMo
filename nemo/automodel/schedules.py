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
from typing import Iterator, Optional, Protocol, Union

import torch
from megatron.core.timers import Timers
from torch.nn.parallel import DistributedDataParallel

from nemo.automodel.llm.causal_lm import AutoModelForCausalLMConfig


class ForwardStepFnProtocol(Protocol):
    def __call__(
        self,
        data_iterator: Iterator,
        model: torch.nn.Module,
        config: AutoModelForCausalLMConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def get_forward_backward_func():
    # TODO: Add pipeline parallelism
    return forward_backward_no_pipelining


def forward_backward_no_pipelining(
    forward_step_func: ForwardStepFnProtocol,
    data_iterator: Union[Iterator, list[Iterator]],
    model: Union[torch.nn.Module, list[torch.nn.Module]],
    config: AutoModelForCausalLMConfig,
    num_microbatches: int,
    forward_only: bool = False,
    timers: Optional[Timers] = None,
):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses.


    See get_forward_backward_func() for argument details
    """

    if isinstance(model, list):
        assert len(model) == 1, "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert len(data_iterator) == 1, "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    if timers is not None:
        timers("forward-backward", log_level=1).start(barrier=config.barrier_with_L1_time)

    if isinstance(model, DistributedDataParallel):
        no_sync_func = model.no_sync
    else:
        no_sync_func = contextlib.nullcontext

    forward_data_store = []
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
    with no_sync_func():
        for i in range(num_microbatches - 1):
            loss, num_tokens = forward_step_func(data_iterator=data_iterator, model=model, config=config)
            num_tokens = num_tokens.clone().detach().to(torch.int)
            total_num_tokens += num_tokens.item()
            forward_data_store.append(loss.clone())
            if not forward_only:
                loss.backward()

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    loss, num_tokens = forward_step_func(data_iterator=data_iterator, model=model, config=config)
    num_tokens = num_tokens.clone().detach().to(torch.int)
    total_num_tokens += num_tokens.item()
    forward_data_store.append(loss.clone())

    if not forward_only:
        loss.backward()

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism and layernorm all-reduce for sequence parallelism).
        config.finalize_model_grads_func(model, total_num_tokens if config.calculate_per_token_loss else None)

    if timers is not None:
        timers("forward-backward").stop()

    return forward_data_store, total_num_tokens
