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

from dataclasses import dataclass
from typing import Optional

import torch
from utils import get_log_probs_y_T_U, get_manifest_lines, make_basetoken_ctm, make_word_ctm

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.parts.utils.transcribe_utils import setup_model
from nemo.core.config import hydra_runner

V_NEG_NUM = -1e30

"""
Align the utterances in manifest_filepath. 
Results are saved in ctm files in output_ctm_folder.

Arguments:
    manifest_filepath: filepath to the manifest of the data you want to align,
        containing 'audio_filepath' and 'text' fields.
    output_ctm_folder: the folder where output CTM files will be saved.
    pretrained_name: string specifying the name of a CTC NeMo ASR model which will be automatically downloaded
        from NGC and used for generating the log-probs which we will use to do alignment.
        Note: NFA can only use CTC models (not Transducer models) at the moment.
    model_path: string specifying the local filepath to a CTC NeMo ASR model which will be used to generate the
        log-probs which we will use to do alignment.
        Note: NFA can only use CTC models (not Transducer models) at the moment.
        Note: if a model_path is provided, it will override the pretrained_name.
    model_downsample_factor: an int indicating the downsample factor of the ASR model, ie the ratio of input 
        timesteps to output timesteps. 
        If the ASR model is a QuartzNet model, its downsample factor is 2.
        If the ASR model is a Conformer CTC model, its downsample factor is 4.
        If the ASR model is a Citirnet model, its downsample factor is 8.
    separator: the string used to separate CTM segments.
        If the separator is “”, the CTM segments will be the tokens used by the ASR model.
        If the separator is anything else, e.g. “ “, “|” or “<new section>”, the segments will be the blocks of 
        text separated by that separator.
        Default: “ “, ie for languages such as English, the CTM segments will be words.
        Note: if the separator is not “” or “ “, it will be removed from the CTM, ie it is treated as a marker 
        which is not part of the ground truth. It will essentially be treated as a space, and any additional spaces 
        around it will be amalgamated into one, i.e. the following texts will be treated as equivalent:
        “abc|def”
        “abc |def”
        “abc| def”
        “abc | def”
    n_parts_for_ctm_id: int specifying how many of the 'parts' of the audio_filepath
        we will use (starting from the final part of the audio_filepath) to determine the 
        utt_id that will be used in the CTM files. Note also that any spaces that are present in the audio_filepath 
        will be stripped away from the utt_id, so as not to change the number of space-separated elements in the 
        CTM files.
        e.g. if audio_filepath is "/a/b/c/d/e 1.wav" and n_parts_for_ctm_id is 1 => utt_id will be "e1"
        e.g. if audio_filepath is "/a/b/c/d/e 1.wav" and n_parts_for_ctm_id is 2 => utt_id will be "d_e1"
        e.g. if audio_filepath is "/a/b/c/d/e 1.wav" and n_parts_for_ctm_id is 3 => utt_id will be "c_d_e1"
    audio_sr: int specifying the sample rate of your audio files.
    device: string specifying the device that will be used for generating log-probs and doing 
        Viterbi decoding. The string needs to be in a format recognized by torch.device()
    batch_size: int specifying batch size that will be used for generating log-probs and doing Viterbi decoding.

"""


@dataclass
class AlignmentConfig:
    # Required configs
    manifest_filepath: Optional[str] = None
    output_ctm_folder: Optional[str] = None

    # General configs
    pretrained_name: Optional[str] = "stt_en_citrinet_1024_gamma_0_25"
    model_path: Optional[str] = None
    model_downsample_factor: int = 8
    separator: str = " "
    n_parts_for_ctm_id: int = 1
    audio_sr: int = 16000
    device: str = "cpu"
    batch_size: int = 1


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


def align_batch(data, model, separator):
    log_probs, y, T, U = get_log_probs_y_T_U(data, model, separator)
    alignments = viterbi_decoding(log_probs, y, T, U)

    return alignments


@hydra_runner(config_name="AlignmentConfig", schema=AlignmentConfig)
def main(cfg: AlignmentConfig):

    # Validate config
    if cfg.manifest_filepath is None:
        raise ValueError("cfg.manifest_filepath needs to be specified")

    if cfg.output_ctm_folder is None:
        raise ValueError("cfg.output_ctm_folder needs to be specified")

    if cfg.model_path is None and cfg.pretrained_name is None:
        raise ValueError("Both cfg.model_path and cfg.pretrained_name cannot be None")

    # load model
    device = torch.device(cfg.device)

    model, _ = setup_model(cfg, device)

    if not isinstance(model, EncDecCTCModel):
        raise NotImplementedError(
            f"Model {cfg.model_name} is not an instance of NeMo EncDecCTCModel."
            " Currently only instances of EncDecCTCModels are supported"
        )

    # define start and end line IDs of batches
    with open(cfg.manifest_filepath, 'r') as f:
        num_lines_in_manifest = sum(1 for _ in f)

    starts = [x for x in range(0, num_lines_in_manifest, cfg.batch_size)]
    ends = [x - 1 for x in starts]
    ends.pop(0)
    ends.append(num_lines_in_manifest)

    # get alignment and save in CTM batch-by-batch
    for start, end in zip(starts, ends):
        data = get_manifest_lines(cfg.manifest_filepath, start, end)

        alignments = align_batch(data, model, cfg.separator)

        if cfg.separator == "":
            make_basetoken_ctm(
                data,
                alignments,
                model,
                cfg.model_downsample_factor,
                cfg.output_ctm_folder,
                cfg.n_parts_for_ctm_id,
                cfg.audio_sr,
            )

        else:
            make_word_ctm(
                data,
                alignments,
                model,
                cfg.model_downsample_factor,
                cfg.output_ctm_folder,
                cfg.n_parts_for_ctm_id,
                cfg.audio_sr,
                cfg.separator,
            )

    return None


if __name__ == "__main__":
    main()
