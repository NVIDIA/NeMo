# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
from utils.constants import V_NEGATIVE_NUM

# rnnt version
def viterbi_decoding(joint, y_batch, T_batch, U_batch, viterbi_device):

    print('joint.shape:', joint.shape)
    print('y_batch.shape:', y_batch.shape)
    print('T_batch:', T_batch)
    print('U_batch', U_batch)


    B, T_max, U_max, V = joint.shape # here V includes blank
    

    # transfer all tensors to viterbi_device
    joint = joint.to(viterbi_device)
    y_batch = y_batch.to(viterbi_device)
    T_batch = T_batch.to(viterbi_device)
    U_batch = U_batch.to(viterbi_device)


    # turn joint into probability using softmax
    log_probs = torch.nn.functional.log_softmax(joint, -1)

    log_probs_only_blank = log_probs[:, :, :, -1]

    y_batch_reshaped = y_batch.unsqueeze(1).unsqueeze(3).repeat(1, T_max, 1, 1)
    print('y_batch_reshaped.shape:', y_batch_reshaped.shape)

    log_probs_ref_text_tokens = torch.gather(
        input = log_probs, 
        dim = -1,
        index = y_batch_reshaped
    )

    print('log_probs_ref_text_tokens.shape:', log_probs_ref_text_tokens.shape)


    viterbi_probs_tensor = V_NEGATIVE_NUM * torch.ones((B, T_max, U_max), device=viterbi_device)
    backpointers_tensor = V_NEGATIVE_NUM * torch.ones((B, T_max, U_max), dtype=torch.int8, device=viterbi_device)

    # initialize first timestep of viterbi_probs_tensor
    viterbi_probs_tensor[:, 0, 0] = log_probs_ref_text_tokens[:, 0, 0]

    for t in range(1, T_max):
        for u in range(U_max):

            if u == 0:
                if t == 0:
                    # base case
                    viterbi_probs_tensor[:, t, u] = 0
                    backpointers_tensor[:, t, u] = -1
                    continue

                else: 
                    # this is case for (t = 0, u > 0), reached by (t, u - 1)
                    # emitting a blank symbol.
                    v_if_emitted_token = V_NEGATIVE_NUM
                    v_if_emitted_blank = viterbi_probs_tensor[:, t-1, u] + log_probs_only_blank[:, t-1, u]
                    
            else:
                if t == 0:
                    # in case of (u > 0, t = 0), this is only reached from
                    # (t, u - 1) with a label emission.
                    v_if_emitted_token = viterbi_probs_tensor[:, t, u-1] + log_probs_ref_text_tokens[:, t, u-1]
                    v_if_emitted_blank = V_NEGATIVE_NUM

                else:
                    v_if_emitted_token = viterbi_probs_tensor[:, t, u-1] + log_probs_ref_text_tokens[:, t, u-1]
                    v_if_emitted_blank = viterbi_probs_tensor[:, t-1, u] + log_probs_only_blank[:, t-1, u]

            if v_if_emitted_token > v_if_emitted_blank:
                new_v = v_if_emitted_token
                backpointer = 0
            
            else:
                new_v = v_if_emitted_blank
                backpointer = 1

            # update viterbi_probs_tensor and backpointers_tensor
            viterbi_probs_tensor[:, t, u] = new_v
            backpointers_tensor[:, t, u] = backpointer

    print('viterbi_probs_tensor:', viterbi_probs_tensor)
    print('backpointers_tensor:', backpointers_tensor)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 10))
    plt.imshow(viterbi_probs_tensor[0][1:].cpu().numpy())
    plt.savefig('viterbi_probs.png')

    plt.figure(figsize=(5, 10))
    plt.imshow(log_probs_ref_text_tokens[0].cpu().numpy())
    plt.savefig('log_probs_ref_text_tokens.png')
    
    import numpy as np
    np.save('log_probs_ref_text_tokens.npy', log_probs_ref_text_tokens[0].cpu().numpy())
    np.save('log_probs_only_blank.npy', log_probs_only_blank[0].cpu().numpy())

    # trace backpointers
    path_tensor = torch.zeros((B, T_max, U_max), dtype=torch.int8, device=viterbi_device)
    path_coords = []

    for b in range(B):
        path_coords_b = []

        t = T_max - 1
        u = U_max - 1

        while t >= 0 and u >= 0:
            path_tensor[b, t, u] = 1
            path_coords_b.append((t, u))

            if backpointers_tensor[b, t, u] == 0:
                u -= 1
            else:
                t -= 1

        path_coords_b.reverse()

        path_coords.append(path_coords_b)



    return path_tensor, path_coords


def viterbi_decoding_ctc(log_probs_batch, y_batch, T_batch, U_batch, viterbi_device):
    """
    Do Viterbi decoding with an efficient algorithm (the only for-loop in the 'forward pass' is over the time dimension). 
    Args:
        log_probs_batch: tensor of shape (B, T_max, V). The parts of log_probs_batch which are 'padding' are filled
            with 'V_NEGATIVE_NUM' - a large negative number which represents a very low probability.
        y_batch: tensor of shape (B, U_max) - contains token IDs including blanks in every other position. The parts of
            y_batch which are padding are filled with the number 'V'. V = the number of tokens in the vocabulary + 1 for
            the blank token.
        T_batch: tensor of shape (B, 1) - contains the durations of the log_probs_batch (so we can ignore the 
            parts of log_probs_batch which are padding)
        U_batch: tensor of shape (B, 1) - contains the lengths of y_batch (so we can ignore the parts of y_batch
            which are padding).
        viterbi_device: the torch device on which Viterbi decoding will be done.

    Returns:
        alignments_batch: list of lists containing locations for the tokens we align to at each timestep.
            Looks like: [[0, 0, 1, 2, 2, 3, 3, ...,  ], ..., [0, 1, 2, 2, 2, 3, 4, ....]].
            Each list inside alignments_batch is of length T_batch[location of utt in batch].
    """

    B, T_max, _ = log_probs_batch.shape
    U_max = y_batch.shape[1]

    # transfer all tensors to viterbi_device
    log_probs_batch = log_probs_batch.to(viterbi_device)
    y_batch = y_batch.to(viterbi_device)
    T_batch = T_batch.to(viterbi_device)
    U_batch = U_batch.to(viterbi_device)

    # make tensor that we will put at timesteps beyond the duration of the audio
    padding_for_log_probs = V_NEGATIVE_NUM * torch.ones((B, T_max, 1), device=viterbi_device)
    # make log_probs_padded tensor of shape (B, T_max, V +1 ) where all of
    # log_probs_padded[:,:,-1] is the 'V_NEGATIVE_NUM'
    log_probs_padded = torch.cat((log_probs_batch, padding_for_log_probs), dim=2)

    # initialize v_prev - tensor of previous timestep's viterbi probabilies, of shape (B, U_max)
    v_prev = V_NEGATIVE_NUM * torch.ones((B, U_max), device=viterbi_device)
    v_prev[:, :2] = torch.gather(input=log_probs_padded[:, 0, :], dim=1, index=y_batch[:, :2])

    # initialize backpointers_rel - which contains values like 0 to indicate the backpointer is to the same u index,
    # 1 to indicate the backpointer pointing to the u-1 index and 2 to indicate the backpointer is pointing to the u-2 index
    backpointers_rel = -99 * torch.ones((B, T_max, U_max), dtype=torch.int8, device=viterbi_device)

    # Make a letter_repetition_mask the same shape as y_batch
    # the letter_repetition_mask will have 'True' where the token (including blanks) is the same
    # as the token two places before it in the ground truth (and 'False everywhere else).
    # We will use letter_repetition_mask to determine whether the Viterbi algorithm needs to look two tokens back or
    # three tokens back
    y_shifted_left = torch.roll(y_batch, shifts=2, dims=1)
    letter_repetition_mask = y_batch - y_shifted_left
    letter_repetition_mask[:, :2] = 1  # make sure dont apply mask to first 2 tokens
    letter_repetition_mask = letter_repetition_mask == 0

    for t in range(1, T_max):

        # e_current is a tensor of shape (B, U_max) of the log probs of every possible token at the current timestep
        e_current = torch.gather(input=log_probs_padded[:, t, :], dim=1, index=y_batch)

        # apply a mask to e_current to cope with the fact that we do not keep the whole v_matrix and continue
        # calculating viterbi probabilities during some 'padding' timesteps
        t_exceeded_T_batch = t >= T_batch

        U_can_be_final = torch.logical_or(
            torch.arange(0, U_max, device=viterbi_device).unsqueeze(0) == (U_batch.unsqueeze(1) - 0),
            torch.arange(0, U_max, device=viterbi_device).unsqueeze(0) == (U_batch.unsqueeze(1) - 1),
        )

        mask = torch.logical_not(torch.logical_and(t_exceeded_T_batch.unsqueeze(1), U_can_be_final,)).long()

        e_current = e_current * mask

        # v_prev_shifted is a tensor of shape (B, U_max) of the viterbi probabilities 1 timestep back and 1 token position back
        v_prev_shifted = torch.roll(v_prev, shifts=1, dims=1)
        # by doing a roll shift of size 1, we have brought the viterbi probability in the final token position to the
        # first token position - let's overcome this by 'zeroing out' the probabilities in the firest token position
        v_prev_shifted[:, 0] = V_NEGATIVE_NUM

        # v_prev_shifted2 is a tensor of shape (B, U_max) of the viterbi probabilities 1 timestep back and 2 token position back
        v_prev_shifted2 = torch.roll(v_prev, shifts=2, dims=1)
        v_prev_shifted2[:, :2] = V_NEGATIVE_NUM  # zero out as we did for v_prev_shifted
        # use our letter_repetition_mask to remove the connections between 2 blanks (so we don't skip over a letter)
        # and to remove the connections between 2 consective letters (so we don't skip over a blank)
        v_prev_shifted2.masked_fill_(letter_repetition_mask, V_NEGATIVE_NUM)

        # we need this v_prev_dup tensor so we can calculated the viterbi probability of every possible
        # token position simultaneously
        v_prev_dup = torch.cat(
            (v_prev.unsqueeze(2), v_prev_shifted.unsqueeze(2), v_prev_shifted2.unsqueeze(2),), dim=2,
        )

        # candidates_v_current are our candidate viterbi probabilities for every token position, from which
        # we will pick the max and record the argmax
        candidates_v_current = v_prev_dup + e_current.unsqueeze(2)
        # we straight away save results in v_prev instead of v_current, so that the variable v_prev will be ready for the
        # next iteration of the for-loop
        v_prev, bp_relative = torch.max(candidates_v_current, dim=2)

        backpointers_rel[:, t, :] = bp_relative

    # trace backpointers
    alignments_batch = []
    for b in range(B):
        T_b = int(T_batch[b])
        U_b = int(U_batch[b])

        if U_b == 1:  # i.e. we put only a blank token in the reference text because the reference text is empty
            current_u = 0  # set initial u to 0 and let the rest of the code block run as usual
        else:
            current_u = int(torch.argmax(v_prev[b, U_b - 2 : U_b])) + U_b - 2
        alignment_b = [current_u]
        for t in range(T_max - 1, 0, -1):
            current_u = current_u - int(backpointers_rel[b, t, current_u])
            alignment_b.insert(0, current_u)
        alignment_b = alignment_b[:T_b]
        alignments_batch.append(alignment_b)

    return alignments_batch
