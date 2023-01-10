# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import torch

V_NEG_NUM = -1e30


def viterbi_decoding(log_probs_batch, y_batch, T_batch, U_batch, viterbi_device):
    """
    Does Viterbi decoding.
    Returns:
        alignments_batch: list of lists containing locations for the tokens we align to at each timestep.
            Looks like: [[0, 0, 1, 2, 2, 3, 3, ... ], [0, 1, 2, 2, 2, 3, 4, ....], ...]
    """
    B, T_max, _ = log_probs_batch.shape
    U_max = y_batch.shape[1]

    # transfer all tensors to device
    log_probs_batch = log_probs_batch.to(viterbi_device)
    y_batch = y_batch.to(viterbi_device)
    T_batch = T_batch.to(viterbi_device)
    U_batch = U_batch.to(viterbi_device)

    padding_for_log_probs = V_NEG_NUM * torch.ones((B, T_max, 1), device=viterbi_device)
    log_probs_padded = torch.cat((log_probs_batch, padding_for_log_probs), dim=2)
    log_probs_reordered = torch.gather(input=log_probs_padded, dim=2, index=y_batch.unsqueeze(1).repeat(1, T_max, 1))

    v_matrix = V_NEG_NUM * torch.ones_like(log_probs_reordered)
    backpointers = -999 * torch.ones_like(v_matrix)
    v_matrix[:, 0, :2] = log_probs_reordered[:, 0, :2]

    y_shifted_left = torch.roll(y_batch, shifts=2, dims=1)
    letter_repetition_mask = y_batch - y_shifted_left
    letter_repetition_mask[:, :2] = 1  # make sure dont apply mask to first 2 tokens
    letter_repetition_mask = letter_repetition_mask == 0

    bp_absolute_template = torch.arange(U_max, device=viterbi_device).unsqueeze(0).repeat(B, 1)

    for t in range(1, T_max):

        e_current = log_probs_reordered[:, t, :]

        v_prev = v_matrix[:, t - 1, :]
        v_prev_shifted = torch.roll(v_prev, shifts=1, dims=1)
        v_prev_shifted[:, 0] = V_NEG_NUM

        v_prev_shifted2 = torch.roll(v_prev, shifts=2, dims=1)
        v_prev_shifted2[:, :2] = V_NEG_NUM
        v_prev_shifted2.masked_fill_(letter_repetition_mask, V_NEG_NUM)

        v_prev_dup = torch.cat(
            (v_prev.unsqueeze(2), v_prev_shifted.unsqueeze(2), v_prev_shifted2.unsqueeze(2),), dim=2,
        )

        candidates_v_current = v_prev_dup + e_current.unsqueeze(2)
        v_current, bp_relative = torch.max(candidates_v_current, dim=2)

        bp_absolute = bp_absolute_template - bp_relative

        v_matrix[:, t, :] = v_current
        backpointers[:, t, :] = bp_absolute

    # trace backpointers TODO: parallelize over batch_size
    alignments_batch = []
    for b in range(B):
        T_b = int(T_batch[b])
        U_b = int(U_batch[b])

        final_state = int(torch.argmax(v_matrix[b, T_b - 1, U_b - 2 : U_b])) + U_b - 2
        alignment_b = [final_state]
        for t in range(T_b - 1, 0, -1):
            alignment_b.insert(0, int(backpointers[b, t, alignment_b[0]]))
        alignments_batch.append(alignment_b)

    return alignments_batch
