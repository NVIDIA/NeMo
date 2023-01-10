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

import json
import os
from pathlib import Path

import soundfile as sf


def _get_utt_id(audio_filepath, audio_filepath_parts_in_utt_id):
    fp_parts = Path(audio_filepath).parts[-audio_filepath_parts_in_utt_id:]
    utt_id = Path("_".join(fp_parts)).stem
    utt_id = utt_id.replace(" ", "-")  # replace any spaces in the filepath with dashes
    return utt_id


def add_t_start_end_to_boundary_info(boundary_info_utt, alignment_utt):
    # first remove boundary_info of any items that are not in the alignment
    # the only items we expect not to be in the alignment are blanks which the alignment chooses to skip
    # we will iterate boundary_info in reverse order for this to make popping the items simple
    s_in_alignment = set(alignment_utt)
    for boundary_info_pointer in range(len(boundary_info_utt) - 1, -1, -1):
        s_in_boundary_info = set(
            range(
                boundary_info_utt[boundary_info_pointer]["s_start"],
                boundary_info_utt[boundary_info_pointer]["s_end"] + 1,
            )
        )
        item_not_in_alignment = True
        for s_ in s_in_boundary_info:
            if s_ in s_in_alignment:
                item_not_in_alignment = False

        if item_not_in_alignment:
            boundary_info_utt.pop(boundary_info_pointer)

    # now updated boundary_info with s_start and s_end
    boundary_info_pointer = 0
    for t, s_at_t in enumerate(alignment_utt):
        if s_at_t == boundary_info_utt[boundary_info_pointer]["s_start"]:
            if "t_start" not in boundary_info_utt[boundary_info_pointer]:
                boundary_info_utt[boundary_info_pointer]["t_start"] = t

        if t < len(alignment_utt) - 1:
            if alignment_utt[t + 1] > boundary_info_utt[boundary_info_pointer]["s_end"]:
                if "t_end" not in boundary_info_utt[boundary_info_pointer]:
                    boundary_info_utt[boundary_info_pointer]["t_end"] = t

                boundary_info_pointer += 1
        else:  # i.e. t == len(alignment) - 1, i.e. we are a the final element in alignment
            # add final t_end if we haven't already
            if "t_end" not in boundary_info_utt[boundary_info_pointer]:
                boundary_info_utt[boundary_info_pointer]["t_end"] = t

        if boundary_info_pointer == len(boundary_info_utt):
            # we have finished populating boundary_info with t_start and t_end,
            # but we might have some final remaining elements (blanks) in the alignment which we dont care about
            # => break, so as not to cause issues trying to access boundary_info[boundary_info_pointer]
            break

    return boundary_info_utt


def make_ctm(
    boundary_info_batch,
    alignments_batch,
    manifest_lines_batch,
    model,
    model_downsample_factor,
    output_dir,
    remove_blank_tokens_from_ctm,
    audio_filepath_parts_in_utt_id,
    minimum_timestamp_duration,
    audio_sr,
):
    """
    Note: assume same order of utts in boundary_info_batch, alignments_batch and manifest_lines_batch
    """
    assert len(boundary_info_batch) == len(alignments_batch) == len(manifest_lines_batch)

    BLANK_TOKEN = "<b>"  # TODO: move outside of function

    os.makedirs(output_dir, exist_ok=True)

    spectrogram_hop_length = model.preprocessor.featurizer.hop_length
    timestep_to_sample_ratio = spectrogram_hop_length * model_downsample_factor

    for boundary_info_utt, alignment_utt, manifest_line in zip(
        boundary_info_batch, alignments_batch, manifest_lines_batch
    ):

        boundary_info_utt = add_t_start_end_to_boundary_info(boundary_info_utt, alignment_utt)

        # get utt_id that will be used for saving CTM file as <utt_id>.ctm
        utt_id = _get_utt_id(manifest_line['audio_filepath'], audio_filepath_parts_in_utt_id)

        # get audio file duration if we will need it later
        if minimum_timestamp_duration > 0:
            with sf.SoundFile(manifest_line["audio_filepath"]) as f:
                audio_file_duration = f.frames / f.samplerate

        with open(os.path.join(output_dir, f"{utt_id}.ctm"), "w") as f_ctm:
            for boundary_info_ in boundary_info_utt:
                text = boundary_info_["text"]
                start_sample = boundary_info_["t_start"] * timestep_to_sample_ratio
                end_sample = (boundary_info_["t_end"] + 1) * timestep_to_sample_ratio - 1

                start_time = start_sample / audio_sr
                end_time = end_sample / audio_sr

                if minimum_timestamp_duration > 0 and minimum_timestamp_duration > end_time - start_time:
                    token_mid_point = (start_time + end_time) / 2
                    start_time = max(token_mid_point - minimum_timestamp_duration / 2, 0)
                    end_time = min(token_mid_point + minimum_timestamp_duration / 2, audio_file_duration)

                if not (text == BLANK_TOKEN and remove_blank_tokens_from_ctm):
                    f_ctm.write(f"{utt_id} 1 {start_time:.2f} {end_time - start_time:.2f} {text}\n")

    return None


def make_new_manifest(
    output_dir,
    original_manifest_filepath,
    additional_ctm_grouping_separator,
    audio_filepath_parts_in_utt_id,
    pred_text_all_lines,
):
    if pred_text_all_lines:
        with open(original_manifest_filepath, 'r') as f:
            num_lines_in_manifest = sum(1 for _ in f)

        if not num_lines_in_manifest == len(pred_text_all_lines):
            raise RuntimeError(
                f"Number of lines in the original manifest ({num_lines_in_manifest}) does not match "
                f"the number of pred_texts we have ({len(pred_text_all_lines)}). Something has gone wrong."
            )

    tgt_manifest_name = str(Path(original_manifest_filepath).stem) + "_with_ctm_paths.json"
    tgt_manifest_filepath = str(Path(output_dir) / tgt_manifest_name)

    with open(original_manifest_filepath, 'r') as fin, open(tgt_manifest_filepath, 'w') as fout:
        for i_line, line in enumerate(fin):
            data = json.loads(line)

            utt_id = _get_utt_id(data["audio_filepath"], audio_filepath_parts_in_utt_id)

            data["token_level_ctm_filepath"] = str(Path(output_dir) / "tokens" / f"{utt_id}.ctm")
            data["word_level_ctm_filepath"] = str(Path(output_dir) / "words" / f"{utt_id}.ctm")

            if additional_ctm_grouping_separator:
                data["additional_segment_level_ctm_filepath"] = str(
                    Path(output_dir) / "additional_segments" / f"{utt_id}.ctm"
                )

            if pred_text_all_lines:
                data['pred_text'] = pred_text_all_lines[i_line]

            new_line = json.dumps(data)

            fout.write(f"{new_line}\n")
