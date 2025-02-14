# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import triton
import triton.language as tl


@triton.jit
def _ngram_advance_triton_kernel(
    vocab_size: "tl.constexpr",
    states_ptr,
    new_states_ptr,
    scores_ptr,
    start_state: int,
    max_order: int,
    backoff_to_states_ptr,
    backoff_weights_ptr,
    state_start_arcs_ptr,
    state_end_arcs_ptr,
    to_states_ptr,
    ilabels_ptr,
    arcs_weights_ptr,
    BLOCK_SIZE: "tl.constexpr",
):
    batch_i = tl.program_id(0)
    cur_state = tl.load(states_ptr + batch_i)

    vocab_offsets = tl.arange(0, BLOCK_SIZE)
    vocab_mask = vocab_offsets < vocab_size
    tl.store(new_states_ptr + batch_i * vocab_size + vocab_offsets, -1, mask=vocab_mask)
    tl.store(scores_ptr + batch_i * vocab_size + vocab_offsets, 0.0, mask=vocab_mask)

    accumulated_backoff = 0.0

    for i in range(max_order):
        tl.debug_barrier()
        start_idx = tl.load(state_start_arcs_ptr + cur_state)
        end_idx = tl.load(state_end_arcs_ptr + cur_state)
        indices = start_idx + vocab_offsets
        mask = indices < end_idx

        cur_ilabels = tl.load(ilabels_ptr + indices, mask=mask)
        cur_weights = tl.load(arcs_weights_ptr + indices, mask=mask)
        cur_to_states = tl.load(to_states_ptr + indices, mask=mask)

        not_final_mask = tl.load(new_states_ptr + batch_i * vocab_size + cur_ilabels, mask=mask, other=0) == -1
        tl.store(
            scores_ptr + batch_i * vocab_size + cur_ilabels,
            cur_weights + accumulated_backoff,
            mask=not_final_mask,
        )
        tl.store(new_states_ptr + batch_i * vocab_size + cur_ilabels, cur_to_states, mask=not_final_mask)

        cur_backoff_weight = tl.load(backoff_weights_ptr + cur_state)
        accumulated_backoff += cur_backoff_weight
        cur_state = tl.load(backoff_to_states_ptr + cur_state).to(states_ptr.dtype.element_ty)


@triton.jit
def _ngram_advance_triton_kernel_v2(
    vocab_size: "tl.constexpr",
    states_ptr,
    new_states_ptr,
    scores_ptr,
    start_state: int,
    max_order: int,
    backoff_to_states_ptr,
    backoff_weights_ptr,
    state_start_arcs_ptr,
    state_end_arcs_ptr,
    to_states_ptr,
    ilabels_ptr,
    arcs_weights_ptr,
    BLOCK_SIZE: "tl.constexpr",
):
    batch_i = tl.program_id(0)
    cur_state = tl.load(states_ptr + batch_i)

    vocab_offsets = tl.arange(0, BLOCK_SIZE)
    vocab_mask = vocab_offsets < vocab_size
    tl.store(new_states_ptr + batch_i * vocab_size + vocab_offsets, -1, mask=vocab_mask)
    tl.store(scores_ptr + batch_i * vocab_size + vocab_offsets, 0.0, mask=vocab_mask)

    accumulated_backoff = 0.0
    start_state_not_processed = True
    while start_state_not_processed:
        tl.debug_barrier()
        start_idx = tl.load(state_start_arcs_ptr + cur_state)
        end_idx = tl.load(state_end_arcs_ptr + cur_state)
        indices = start_idx + vocab_offsets
        mask = indices < end_idx

        cur_ilabels = tl.load(ilabels_ptr + indices, mask=mask)
        cur_weights = tl.load(arcs_weights_ptr + indices, mask=mask)
        cur_to_states = tl.load(to_states_ptr + indices, mask=mask)

        not_final_mask = tl.load(new_states_ptr + batch_i * vocab_size + cur_ilabels, mask=mask, other=0) == -1
        tl.store(
            scores_ptr + batch_i * vocab_size + cur_ilabels,
            cur_weights + accumulated_backoff,
            mask=not_final_mask,
        )
        tl.store(new_states_ptr + batch_i * vocab_size + cur_ilabels, cur_to_states, mask=not_final_mask)

        start_state_not_processed = cur_state != start_state
        # backoff
        cur_backoff_weight = tl.load(backoff_weights_ptr + cur_state)
        accumulated_backoff += cur_backoff_weight
        cur_state = tl.load(backoff_to_states_ptr + cur_state).to(states_ptr.dtype.element_ty)
