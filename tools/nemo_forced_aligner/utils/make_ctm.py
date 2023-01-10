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

import os
from pathlib import Path

import soundfile as sf


def _get_utt_id(audio_filepath, n_parts_for_ctm_id):
    fp_parts = Path(audio_filepath).parts[-n_parts_for_ctm_id:]
    utt_id = Path("_".join(fp_parts)).stem
    utt_id = utt_id.replace(" ", "-")  # replace any spaces in the filepath with dashes
    return utt_id


def add_t_start_end_to_boundary_info(boundary_info, alignment):
    # first remove boundary_info of any items that are not in the alignment
    # the only items we expect not to be in the alignment are blanks which the alignment chooses to skip
    # we will iterate boundary_info in reverse order for this to make popping the items simple
    s_in_alignment = set(alignment)
    for boundary_info_pointer in range(len(boundary_info) - 1, -1, -1):
        s_in_boundary_info = set(
            range(boundary_info[boundary_info_pointer]["s_start"], boundary_info[boundary_info_pointer]["s_end"] + 1)
        )
        item_not_in_alignment = True
        for s_ in s_in_boundary_info:
            if s_ in s_in_alignment:
                item_not_in_alignment = False

        if item_not_in_alignment:
            boundary_info.pop(boundary_info_pointer)

    # now updated boundary_info with s_start and s_end
    boundary_info_pointer = 0
    for t, s_at_t in enumerate(alignment):
        if s_at_t == boundary_info[boundary_info_pointer]["s_start"]:
            if "t_start" not in boundary_info[boundary_info_pointer]:
                boundary_info[boundary_info_pointer]["t_start"] = t

        if t < len(alignment) - 1:
            if alignment[t + 1] > boundary_info[boundary_info_pointer]["s_end"]:
                if "t_end" not in boundary_info[boundary_info_pointer]:
                    boundary_info[boundary_info_pointer]["t_end"] = t

                boundary_info_pointer += 1
        else:  # i.e. t == len(alignment) - 1, i.e. we are a the final element in alignment
            # add final t_end if we haven't already
            if "t_end" not in boundary_info[boundary_info_pointer]:
                boundary_info[boundary_info_pointer]["t_end"] = t

        if boundary_info_pointer == len(boundary_info):
            # we have finished populating boundary_info with t_start and t_end,
            # but we might have some final remaining elements (blanks) in the alignment which we dont care about
            # => break, so as not to cause issues trying to access boundary_info[boundary_info_pointer]
            break

    return boundary_info


def make_ctm(
    boundary_info_list,
    alignment_list,
    data_list,
    model,
    model_downsample_factor,
    output_ctm_folder,
    remove_blank_tokens_from_ctm,
    n_parts_for_ctm_id,
    minimum_timestamp_duration,
    audio_sr,
):
    """
    Note: assume same order of utts in boundary_info, alignments and data
    """
    assert len(boundary_info_list) == len(alignment_list) == len(data_list)

    BLANK_TOKEN = "<b>"  # TODO: move outside of function

    os.makedirs(output_ctm_folder, exist_ok=True)

    spectrogram_hop_length = model.preprocessor.featurizer.hop_length
    timestep_to_sample_ratio = spectrogram_hop_length * model_downsample_factor

    for boundary_info, alignment, manifest_line in zip(boundary_info_list, alignment_list, data_list):

        boundary_info = add_t_start_end_to_boundary_info(boundary_info, alignment)

        # get utt_id that will be used for saving CTM file as <utt_id>.ctm
        utt_id = _get_utt_id(manifest_line['audio_filepath'], n_parts_for_ctm_id)

        # get audio file duration if we will need it later
        if minimum_timestamp_duration > 0:
            with sf.SoundFile(manifest_line["audio_filepath"]) as f:
                audio_file_duration = f.frames / f.samplerate

        with open(os.path.join(output_ctm_folder, f"{utt_id}.ctm"), "w") as f_ctm:
            for segment_info in boundary_info:
                text = segment_info["text"]
                start_sample = segment_info["t_start"] * timestep_to_sample_ratio
                end_sample = (segment_info["t_end"] + 1) * timestep_to_sample_ratio - 1

                start_time = start_sample / audio_sr
                end_time = end_sample / audio_sr

                if minimum_timestamp_duration > 0 and minimum_timestamp_duration > end_time - start_time:
                    token_mid_point = (start_time + end_time) / 2
                    start_time = max(token_mid_point - minimum_timestamp_duration / 2, 0)
                    end_time = min(token_mid_point + minimum_timestamp_duration / 2, audio_file_duration)

                if not (text == BLANK_TOKEN and remove_blank_tokens_from_ctm):
                    f_ctm.write(f"{utt_id} 1 {start_time:.5f} {end_time - start_time:.5f} {text}\n")

    return None
