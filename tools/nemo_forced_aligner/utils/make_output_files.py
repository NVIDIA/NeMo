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

import json
import os
from pathlib import Path

import soundfile as sf
from utils.constants import BLANK_TOKEN, SPACE_TOKEN
from utils.data_prep import Token, Word, Segment, Utterance


def _get_utt_id(audio_filepath, audio_filepath_parts_in_utt_id):
    fp_parts = Path(audio_filepath).parts[-audio_filepath_parts_in_utt_id:]
    utt_id = Path("_".join(fp_parts)).stem
    utt_id = utt_id.replace(" ", "-")  # replace any spaces in the filepath with dashes
    return utt_id


def add_t_start_end_to_utt_obj(utt_obj, alignment_utt):
    """
    TODO
    """

    # General idea:
    # t_start is the location of the first appearance of s_start in alignment_utt
    # t_end is the location of the final appearance of s_end in alignment_utt
    # We will make dictionaries num_to_first_alignment_appearance and
    # num_to_last_appearance and use that to update all of
    # the t_start and t_end values in utt_obj.
    # We will put t_start = t_end = -1 for tokens that are skipped (should only be blanks)

    num_to_first_alignment_appearance = dict()
    num_to_last_alignment_appearance = dict()

    prev_s = -1  # use prev_s to keep track of when the s changes
    for t, s in enumerate(alignment_utt):
        if s > prev_s:
            num_to_first_alignment_appearance[s] = t

            if prev_s >= 0:  # dont record prev_s = -1
                num_to_last_alignment_appearance[prev_s] = t - 1
        prev_s = s
    # add last appearance of the final s
    num_to_last_alignment_appearance[prev_s] = len(alignment_utt) - 1

    # update all the t_start and t_end in utt_obj
    for segment_or_token in utt_obj.segments_and_tokens:
        if type(segment_or_token) is Segment:
            segment = segment_or_token
            segment.t_start = num_to_first_alignment_appearance[segment.s_start]
            segment.t_end = num_to_last_alignment_appearance[segment.s_end]

            for word_or_token in segment.words_and_tokens:
                if type(word_or_token) is Word:
                    word = word_or_token
                    word.t_start = num_to_first_alignment_appearance[word.s_start]
                    word.t_end = num_to_last_alignment_appearance[word.s_end]

                    for token in word.tokens:
                        token.t_start = num_to_first_alignment_appearance.get(token.s_start, -1)
                        token.t_end = num_to_last_alignment_appearance.get(token.s_end, -1)
                else:
                    token = word_or_token
                    token.t_start = num_to_first_alignment_appearance.get(token.s_start, -1)
                    token.t_end = num_to_last_alignment_appearance.get(token.s_end, -1)

        else:
            token = segment_or_token
            token.t_start = num_to_first_alignment_appearance.get(token.s_start, -1)
            token.t_end = num_to_last_alignment_appearance.get(token.s_end, -1)

    return utt_obj


def make_ctms(
    utt_obj_batch,
    alignments_batch,
    manifest_lines_batch,
    model,
    model_downsample_factor,
    output_dir_root,
    remove_blank_tokens_from_ctm,
    audio_filepath_parts_in_utt_id,
    minimum_timestamp_duration,
):
    """
    Function to save CTM files for all the utterances in the incoming batch.
    """

    assert len(utt_obj_batch) == len(alignments_batch) == len(manifest_lines_batch)
    # we also assume that utterances are in the same order in utt_obj_batch, alignments_batch
    # and manifest_lines_batch - this should be the case unless there is a strange bug upstream in the
    # code

    # the ratio to convert from timesteps (the units of 't_start' and 't_end' in boundary_info_utt)
    # to the number of samples ('samples' in the sense of 16000 'samples' per second)
    timestep_to_sample_ratio = model.preprocessor.featurizer.hop_length * model_downsample_factor

    for utt_obj, alignment_utt, manifest_line in zip(utt_obj_batch, alignments_batch, manifest_lines_batch):

        utt_obj = add_t_start_end_to_utt_obj(utt_obj, alignment_utt)

        # get utt_id that will be used for saving CTM file as <utt_id>.ctm
        utt_id = _get_utt_id(manifest_line['audio_filepath'], audio_filepath_parts_in_utt_id)

        # get audio file duration if we will need it later
        if minimum_timestamp_duration > 0:
            with sf.SoundFile(manifest_line["audio_filepath"]) as f:
                audio_file_duration = f.frames / f.samplerate
        else:
            audio_file_duration = None

        make_ctm(
            "tokens",
            utt_id,
            utt_obj,
            manifest_lines_batch,
            model,
            timestep_to_sample_ratio,
            output_dir_root,
            remove_blank_tokens_from_ctm,
            audio_filepath_parts_in_utt_id,
            minimum_timestamp_duration,
            audio_file_duration,
        )

        make_ctm(
            "words",
            utt_id,
            utt_obj,
            manifest_lines_batch,
            model,
            timestep_to_sample_ratio,
            output_dir_root,
            remove_blank_tokens_from_ctm,
            audio_filepath_parts_in_utt_id,
            minimum_timestamp_duration,
            audio_file_duration,
        )

        make_ctm(
            "segments",
            utt_id,
            utt_obj,
            manifest_lines_batch,
            model,
            timestep_to_sample_ratio,
            output_dir_root,
            remove_blank_tokens_from_ctm,
            audio_filepath_parts_in_utt_id,
            minimum_timestamp_duration,
            audio_file_duration,
        )

    return None


def make_ctm(
    alignment_level,
    utt_id,
    utt_obj,
    manifest_lines_batch,
    model,
    timestep_to_sample_ratio,
    output_dir_root,
    remove_blank_tokens_from_ctm,
    audio_filepath_parts_in_utt_id,
    minimum_timestamp_duration,
    audio_file_duration,
):
    output_dir = os.path.join(output_dir_root, alignment_level)

    os.makedirs(output_dir, exist_ok=True)

    boundary_info_utt = []

    for segment_or_token in utt_obj.segments_and_tokens:
        if type(segment_or_token) is Segment:
            segment = segment_or_token
            if alignment_level == "segments":
                boundary_info_utt.append(segment)

            for word_or_token in segment.words_and_tokens:
                if type(word_or_token) is Word:
                    word = word_or_token
                    if alignment_level == "words":
                        boundary_info_utt.append(word)

                    for token in word.tokens:
                        if alignment_level == "tokens":
                            boundary_info_utt.append(token)

                else:
                    token = word_or_token
                    if alignment_level == "tokens":
                        boundary_info_utt.append(token)

        else:
            token = segment_or_token
            if alignment_level == "tokens":
                boundary_info_utt.append(token)

    with open(os.path.join(output_dir, f"{utt_id}.ctm"), "w") as f_ctm:
        for boundary_info_ in boundary_info_utt:  # loop over every token/word/segment

            # skip if t_start = t_end = -1 because we used it as a marker to skip some blank tokens
            if not (boundary_info_.t_start == -1 or boundary_info_.t_end == -1):
                text = boundary_info_.text
                start_sample = boundary_info_.t_start * timestep_to_sample_ratio
                end_sample = (boundary_info_.t_end + 1) * timestep_to_sample_ratio - 1

                if not 'sample_rate' in model.cfg.preprocessor:
                    raise ValueError(
                        "Don't have attribute 'sample_rate' in 'model.cfg.preprocessor' => cannot calculate start "
                        " and end time of segments => stopping process"
                    )

                start_time = start_sample / model.cfg.preprocessor.sample_rate
                end_time = end_sample / model.cfg.preprocessor.sample_rate

                if minimum_timestamp_duration > 0 and minimum_timestamp_duration > end_time - start_time:
                    # make the predicted duration of the token/word/segment longer, growing it outwards equal
                    # amounts from the predicted center of the token/word/segment
                    token_mid_point = (start_time + end_time) / 2
                    start_time = max(token_mid_point - minimum_timestamp_duration / 2, 0)
                    end_time = min(token_mid_point + minimum_timestamp_duration / 2, audio_file_duration)

                if not (text == BLANK_TOKEN and remove_blank_tokens_from_ctm):  # don't save blanks if we don't want to
                    # replace any spaces with <space> so we dont introduce extra space characters to our CTM files
                    text = text.replace(" ", SPACE_TOKEN)

                    f_ctm.write(f"{utt_id} 1 {start_time:.2f} {end_time - start_time:.2f} {text}\n")

    return None


def make_new_manifest(
    output_dir,
    original_manifest_filepath,
    additional_ctm_grouping_separator,
    audio_filepath_parts_in_utt_id,
    pred_text_all_lines,
):
    """
    Function to save a new manifest with the same info as the original manifest, but also the paths to the
    CTM files for each utterance and the "pred_text" if it was used for the alignment.
    """
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
            data["segment_level_ctm_filepath"] = str(Path(output_dir) / "segments" / f"{utt_id}.ctm")

            if pred_text_all_lines:
                data['pred_text'] = pred_text_all_lines[i_line]

            # change ctm filepaths to None if they do not actually exits (which will happen if text was empty)
            for key in ["token_level_ctm_filepath", "word_level_ctm_filepath", "segment_level_ctm_filepath"]:
                if not os.path.exists(key):
                    data[key] = None

            new_line = json.dumps(data)

            fout.write(f"{new_line}\n")
