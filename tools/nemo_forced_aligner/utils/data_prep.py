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

import soundfile as sf
import torch

V_NEG_NUM = -1e30


def get_batch_starts_ends(manifest_filepath, batch_size):
    """
    Get the start and end ids of the lines we will use for each 'batch'.
    """

    with open(manifest_filepath, 'r') as f:
        num_lines_in_manifest = sum(1 for _ in f)

    starts = [x for x in range(0, num_lines_in_manifest, batch_size)]
    ends = [x - 1 for x in starts]
    ends.pop(0)
    ends.append(num_lines_in_manifest)

    return starts, ends


def is_entry_in_any_lines(manifest_filepath, entry):
    """
    Returns True if entry is a key in any of the JSON lines in manifest_filepath
    """

    entry_in_manifest = False

    with open(manifest_filepath, 'r') as f:
        for line in f:
            data = json.loads(line)

            if entry in data:
                entry_in_manifest = True

    return entry_in_manifest


def is_entry_in_all_lines(manifest_filepath, entry):
    """
    Returns True is entry is a key in all of the JSON lines in manifest_filepath.
    """
    with open(manifest_filepath, 'r') as f:
        for line in f:
            data = json.loads(line)

            if entry not in data:
                return False

    return True


def get_audio_sr(manifest_filepath):
    """
    Measure the sampling rate of the audio file in the first line
    of the manifest at manifest_filepath
    """
    with open(manifest_filepath, "r") as f_manifest:
        first_line = json.loads(f_manifest.readline())

    audio_file = first_line["audio_filepath"]
    if not os.path.exists(audio_file):
        raise RuntimeError(f"Did not find filepath {audio_file} which was specified in manifest {manifest_filepath}.")

    with sf.SoundFile(audio_file, "r") as f_audio:
        return f_audio.samplerate


def get_manifest_lines(manifest_filepath, start, end):
    data = []
    with open(manifest_filepath, "r") as f:
        for line_i, line in enumerate(f):
            if line_i == start and line_i == end:
                data.append(json.loads(line))
                break

            if line_i == end:
                break
            if line_i >= start:
                data.append(json.loads(line))
    return data


def get_char_tokens(text, model):
    tokens = []
    for character in text:
        if character in model.decoder.vocabulary:
            tokens.append(model.decoder.vocabulary.index(character))
        else:
            tokens.append(len(model.decoder.vocabulary))  # return unk token (same as blank token)

    return tokens


def get_y_token_word_segment_info(text, model, separator):

    if hasattr(model, 'tokenizer'):

        if not separator:  # if separator is not defined - treat the whole text as one segment
            segments = [text]
        else:
            segments = text.split(separator)

        # remove any spaces at start and end of segments
        segments = [seg.strip() for seg in segments]

        BLANK_ID = len(model.decoder.vocabulary)  # TODO: check
        BLANK_TOKEN = "<blank>"
        y_token_ids = []
        y_token_ids_with_blanks = [BLANK_ID]
        y_tokens = []
        y_tokens_with_blanks = [BLANK_TOKEN]
        token_info = [{"text": BLANK_TOKEN, "s_start": 0, "s_end": 0,}]
        word_info = []
        segment_info = []

        segment_s_pointer = 1
        word_s_pointer = 1

        for segment in segments:
            words = segment.split(" ")
            for word in words:

                word_tokens = model.tokenizer.text_to_tokens(word)
                word_ids = model.tokenizer.text_to_ids(word)
                for token, id_ in zip(word_tokens, word_ids):
                    y_token_ids.append(id_)
                    y_token_ids_with_blanks.extend([id_, BLANK_ID])
                    y_tokens.append(token)
                    y_tokens_with_blanks.extend([token, BLANK_TOKEN])

                    token_info.append(
                        {
                            "text": token,
                            "s_start": len(y_tokens_with_blanks) - 2,
                            "s_end": len(y_tokens_with_blanks) - 2,
                        }
                    )
                    token_info.append(
                        {
                            "text": BLANK_TOKEN,
                            "s_start": len(y_tokens_with_blanks) - 1,
                            "s_end": len(y_tokens_with_blanks) - 1,
                        }
                    )

                word_info.append(
                    {
                        "text": word,
                        "s_start": word_s_pointer,
                        "s_end": word_s_pointer + (len(word_tokens) - 1) * 2,  # TODO check this,
                    }
                )
                word_s_pointer += len(word_tokens) * 2  # TODO check this

            segment_tokens = model.tokenizer.text_to_tokens(segment)
            segment_info.append(
                {
                    "text": segment,
                    "s_start": segment_s_pointer,
                    "s_end": segment_s_pointer + (len(segment_tokens) - 1) * 2,
                }
            )
            segment_s_pointer += len(segment_tokens) * 2

        return y_token_ids_with_blanks, token_info, word_info, segment_info

    elif hasattr(model.decoder, "vocabulary"):

        segments = text.split(separator)

        # remove any spaces at start and end of segments
        segments = [seg.strip() for seg in segments]

        BLANK_ID = len(model.decoder.vocabulary)  # TODO: check this is correct
        BLANK_TOKEN = "<b>"
        SPACE_ID = model.decoder.vocabulary.index(" ")
        SPACE_TOKEN = "<space>"
        y_token_ids = []
        y_token_ids_with_blanks = [BLANK_ID]
        y_tokens = []
        y_tokens_with_blanks = [BLANK_TOKEN]
        token_info = [{"text": BLANK_TOKEN, "s_start": 0, "s_end": 0,}]
        word_info = []
        segment_info = []

        segment_s_pointer = 1
        word_s_pointer = 1

        for i_segment, segment in enumerate(segments):
            words = segment.split(" ")
            for i_word, word in enumerate(words):

                word_tokens = list(word)
                word_ids = get_char_tokens(word, model)
                for token, id_ in zip(word_tokens, word_ids):
                    y_token_ids.append(id_)
                    y_token_ids_with_blanks.extend([id_, BLANK_ID])
                    y_tokens.append(token)
                    y_tokens_with_blanks.extend([token, BLANK_TOKEN])

                    token_info.append(
                        {
                            "text": token,
                            "s_start": len(y_tokens_with_blanks) - 2,
                            "s_end": len(y_tokens_with_blanks) - 2,
                        }
                    )
                    token_info.append(
                        {
                            "text": BLANK_TOKEN,
                            "s_start": len(y_tokens_with_blanks) - 1,
                            "s_end": len(y_tokens_with_blanks) - 1,
                        }
                    )

                # add space token unless this is the final word in the final segment
                if not (i_segment == len(segments) - 1 and i_word == len(words) - 1):
                    y_token_ids.append(SPACE_ID)
                    y_token_ids_with_blanks.extend([SPACE_ID, BLANK_ID])
                    y_tokens.append(SPACE_TOKEN)
                    y_tokens_with_blanks.extend([SPACE_TOKEN, BLANK_TOKEN])
                    token_info.append(
                        {
                            "text": SPACE_TOKEN,
                            "s_start": len(y_tokens_with_blanks) - 2,
                            "s_end": len(y_tokens_with_blanks) - 2,
                        }
                    )
                    token_info.append(
                        {
                            "text": BLANK_TOKEN,
                            "s_start": len(y_tokens_with_blanks) - 1,
                            "s_end": len(y_tokens_with_blanks) - 1,
                        }
                    )
                word_info.append(
                    {
                        "text": word,
                        "s_start": word_s_pointer,
                        "s_end": word_s_pointer + len(word_tokens) * 2 - 2,  # TODO check this,
                    }
                )
                word_s_pointer += len(word_tokens) * 2 + 2  # TODO check this

            segment_tokens = get_char_tokens(segment, model)
            segment_info.append(
                {
                    "text": segment,
                    "s_start": segment_s_pointer,
                    "s_end": segment_s_pointer + (len(segment_tokens) - 1) * 2,
                }
            )
            segment_s_pointer += len(segment_tokens) * 2 + 2

        return y_token_ids_with_blanks, token_info, word_info, segment_info

    else:
        raise RuntimeError("Cannot get tokens of this model.")


def get_log_probs_y_T_U(data, model, separator, align_using_pred_text):
    """
    Preparing some tensors to be used during Viterbi decoding.
    Returns:
        log_probs, y, T, U (y and U are s.t. every other token is a blank),
        token_info_list, word_info_list, segment_info_list,
        pred_text_list
    """

    audio_filepaths = [line["audio_filepath"] for line in data]
    B = len(audio_filepaths)

    with torch.no_grad():
        hypotheses = model.transcribe(audio_filepaths, return_hypotheses=True, batch_size=B)

    log_probs_list = []
    T_list = []
    pred_text_list = []
    for hypothesis in hypotheses:
        log_probs_list.append(hypothesis.y_sequence)
        T_list.append(hypothesis.y_sequence.shape[0])
        pred_text_list.append(hypothesis.text)

    y_list = []
    U_list = []
    token_info_list = []
    word_info_list = []
    segment_info_list = []

    for i_line, line in enumerate(data):
        if align_using_pred_text:
            gt_text_for_alignment = pred_text_list[i_line]
        else:
            gt_text_for_alignment = line["text"]
        y_line, token_info, word_info, segment_info = get_y_token_word_segment_info(
            gt_text_for_alignment, model, separator
        )

        y_list.append(y_line)
        U_list.append(len(y_line))
        token_info_list.append(token_info)
        word_info_list.append(word_info)
        segment_info_list.append(segment_info)

    T_max = max(T_list)
    U_max = max(U_list)
    V = len(model.decoder.vocabulary) + 1
    T = torch.tensor(T_list)
    U = torch.tensor(U_list)

    # make log_probs tensor of shape (B x T_max x V)
    log_probs = V_NEG_NUM * torch.ones((B, T_max, V))
    for b, log_probs_b in enumerate(log_probs_list):
        t = log_probs_b.shape[0]
        log_probs[b, :t, :] = log_probs_b

    # make y tensor of shape (B x U_max)
    y = V * torch.ones((B, U_max), dtype=torch.int64)
    for b, y_b in enumerate(y_list):
        U_b = U[b]
        y[b, :U_b] = torch.tensor(y_b)

    return log_probs, y, T, U, token_info_list, word_info_list, segment_info_list, pred_text_list
