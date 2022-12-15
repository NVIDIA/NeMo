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
from utils.data_prep import get_log_probs_y_T_U, get_manifest_lines
from utils.make_ctm import make_basetoken_ctm, make_word_ctm
from utils.viterbi_decoding import viterbi_decoding

from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.parts.utils.transcribe_utils import setup_model
from nemo.core.config import hydra_runner


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
        will be replaced with dashes, so as not to change the number of space-separated elements in the 
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

        log_probs, y, T, U = get_log_probs_y_T_U(data, model, cfg.separator)
        alignments = viterbi_decoding(log_probs, y, T, U)

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
