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

from pathlib import Path

import torch
from utils import get_log_probs_y_T_U, get_manifest_lines, make_basetoken_ctm, make_word_ctm

from nemo.collections.asr.models import ASRModel

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
    B, T_max, V = log_probs.shape
    U_max = y.shape[1]

    padding_for_log_probs = V_NEG_NUM * torch.ones((B, T_max, 1))
    log_probs_padded = torch.cat((log_probs, padding_for_log_probs), dim=2)
    log_probs_reordered = torch.gather(input=log_probs_padded, dim=2, index=y.unsqueeze(1).repeat(1, T_max, 1))
    log_probs_reordered = log_probs_reordered.cpu()  # TODO: do alignment on GPU if available

    v_matrix = V_NEG_NUM * torch.ones_like(log_probs_reordered)
    backpointers = -999 * torch.ones_like(v_matrix)
    v_matrix[:, 0, :2] = log_probs_reordered[:, 0, :2]

    y_shifted_left = torch.roll(y, shifts=2, dims=1)
    letter_repetition_mask = y - y_shifted_left
    letter_repetition_mask[:, :2] = 1  # make sure dont apply mask to first 2 tokens
    letter_repetition_mask = letter_repetition_mask == 0

    for t in range(1, T_max):

        e_current = log_probs_reordered[:, t, :]

        v_prev = v_matrix[:, t - 1, :]
        v_prev_shifted = torch.roll(v_prev, shifts=1, dims=1)
        v_prev_shifted[:, 0] = V_NEG_NUM

        v_prev_shifted2 = torch.roll(v_prev, shifts=2, dims=1)
        v_prev_shifted2[:, :2] = V_NEG_NUM
        v_prev_shifted2 = v_prev_shifted2.masked_fill(letter_repetition_mask, V_NEG_NUM)  # TODO: do in-place instead?

        v_prev_dup = torch.cat(
            (v_prev.unsqueeze(2), v_prev_shifted.unsqueeze(2), v_prev_shifted2.unsqueeze(2),), dim=2,
        )

        candidates_v_current = v_prev_dup + e_current.unsqueeze(2)
        v_current, bp_relative = torch.max(candidates_v_current, dim=2)

        bp_absolute = torch.arange(U_max).unsqueeze(0).repeat(B, 1) - bp_relative

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

    return alignments, v_matrix, log_probs_reordered


def align_batch(data, model):
    log_probs, y, T, U = get_log_probs_y_T_U(data, model)
    alignments, v_matrix, log_probs_reordered = viterbi_decoding(log_probs, y, T, U)

    return alignments, v_matrix, log_probs_reordered


def align(
    manifest_filepath,
    model_name,
    model_downsample_factor,
    output_ctm_folder,
    grouping_for_ctm,
    utt_id_extractor_func=lambda fp: Path(fp).resolve().stem,
    audio_sr=16000,  # TODO: get audio SR automatically
    device="cuda:0",
    batch_size=1,
):
    """
    Function that does alignment of utterances in `manifest_filepath`.
    Results are saved in ctm files in `output_ctm_folder`.
    Returns the most recent alignment, v_matrix, log_probs_reordered, in case you want to inspect them
    in a parent function.

    TODO: add check that Model is CTC/CTCBPE model
    """

    # load model
    device = torch.device(device)
    if ".nemo" in model_name:
        model = ASRModel.restore_from(model_name, map_location=device)
    else:
        model = ASRModel.from_pretrained(model_name, map_location=device)

    # define start and end line IDs of batches
    num_lines_in_manifest = sum(1 for _ in open(manifest_filepath))
    starts = [x for x in range(0, num_lines_in_manifest, batch_size)]
    ends = [x - 1 for x in starts]
    ends.pop(0)
    ends.append(num_lines_in_manifest)

    # get alignment and save in CTM batch-by-batch
    for start, end in zip(starts, ends):
        data = get_manifest_lines(manifest_filepath, start, end)

        alignments, v_matrix, log_probs_reordered = align_batch(data, model)

        if grouping_for_ctm == "basetoken":
            make_basetoken_ctm(
                data, alignments, model, model_downsample_factor, output_ctm_folder, utt_id_extractor_func, audio_sr,
            )

        elif grouping_for_ctm == "word":
            make_word_ctm(
                data, alignments, model, model_downsample_factor, output_ctm_folder, utt_id_extractor_func, audio_sr,
            )

        else:
            raise ValueError(f"Unexpected value for grouping_for_ctm: {grouping_for_ctm}")

    return alignments, v_matrix, log_probs_reordered
