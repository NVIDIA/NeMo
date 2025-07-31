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

from functools import partial
from typing import Iterable

from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.utils import get_batch_on_this_cp_rank

from nemo.tron.config import ConfigContainer, FinetuningDatasetConfig
from nemo.tron.llm.utils import (get_batch_from_iterator,
                                 get_batch_on_this_tp_rank)
from nemo.tron.losses import masked_next_token_loss
from nemo.tron.state import GlobalState


def get_batch(data_iterator, cfg: ConfigContainer):
    """Generate a batch."""

    if (not parallel_state.is_pipeline_first_stage()) and (not parallel_state.is_pipeline_last_stage()):
        return None, None, None, None, None

    if isinstance(cfg.dataset_config, FinetuningDatasetConfig):
        batch = get_batch_from_iterator(data_iterator)
    else:
        # get batches based on the TP rank you are on
        batch = get_batch_on_this_tp_rank(data_iterator, cfg)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


def forward_step(state: GlobalState, data_iterator: Iterable, model: GPTModel):
    """Forward training step.

    Args:
        state (GlobalState): Global state for the run
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """

    timers = state.timers
    straggler_timer = state.straggler_timer

    timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator, state.cfg)
    timers("batch-generator").stop()

    with straggler_timer:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(masked_next_token_loss, loss_mask)
