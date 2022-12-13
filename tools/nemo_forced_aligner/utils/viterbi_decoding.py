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


def viterbi_decoding(log_probs, y, T, U):
    """
    Does Viterbi decoding.
    Returns:
        alignments: list of lists containing locations for the tokens we align to at each timestep
            looks like: [[0, 0, 1, 2, 2, 3, 3, ... ], [0, 1, 2, 2, 2, 3, 4, ....], ...]
        v_matrix: 
            tensor of shape (Batch, Time, Length_of_U_including_interspersed_blanks) 
            containing viterbi probabilities
            This is returned in case you want to examine it in a parent function
        log_probs_reordered: 
            log_probs reordered to match the order of ground truth tokens
            tensor of shape (Batch, Time, Length_of_U_including_interspersed_blanks) 
            This is returned in case you want to examine it in a parent function
    """
    B, T_max, _ = log_probs.shape
    U_max = y.shape[1]

    device = log_probs.device

    padding_for_log_probs = V_NEG_NUM * torch.ones((B, T_max, 1), device=device)
    log_probs_padded = torch.cat((log_probs, padding_for_log_probs), dim=2)
    log_probs_reordered = torch.gather(input=log_probs_padded, dim=2, index=y.unsqueeze(1).repeat(1, T_max, 1))

    v_matrix = V_NEG_NUM * torch.ones_like(log_probs_reordered)
    backpointers = -999 * torch.ones_like(v_matrix)
    v_matrix[:, 0, :2] = log_probs_reordered[:, 0, :2]

    y_shifted_left = torch.roll(y, shifts=2, dims=1)
    letter_repetition_mask = y - y_shifted_left
    letter_repetition_mask[:, :2] = 1  # make sure dont apply mask to first 2 tokens
    letter_repetition_mask = letter_repetition_mask == 0

    bp_absolute_template = torch.arange(U_max, device=device).unsqueeze(0).repeat(B, 1)

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
    alignments = []
    for b in range(B):
        T_b = int(T[b])
        U_b = int(U[b])

        final_state = int(torch.argmax(v_matrix[b, T_b - 1, U_b - 2 : U_b])) + U_b - 2
        alignment_b = [final_state]
        for t in range(T_b - 1, 0, -1):
            alignment_b.insert(0, int(backpointers[b, t, alignment_b[0]]))
        alignments.append(alignment_b)

    return alignments
