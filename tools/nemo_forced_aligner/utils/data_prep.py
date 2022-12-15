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


def get_audio_sr(manifest_filepath):
    """
    Measure the sampling rate of the audio file in the first line
    of the manifest at manifest_filepath
    """
    with open(manifest_filepath, "r") as f_manifest:
        first_line = json.loads(f_manifest.readline())

    audio_file = first_line["audio_filepath"]
    if not os.path.exists(audio_file):
        raise RuntimeError(
            f"Did not find filepath {audio_file} which was specified in manifest {manifest_filepath}."
        )

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


def get_processed_text_and_segments(text, separator):
    """
    If the separator is not empty string, ie CTM segments will not just be tokens,
    this function replaces the separator with a space, amalgamates any spaces around
    it into one, and returns the new processed text as well as a list of strings
    containing the segments to be returned in the CTM.
    """
    if separator == "":
        return text, None

    segments = text.split(separator)

    # remove any spaces at start and end of segments
    segments = [seg.strip() for seg in segments]

    processed_text = " ".join(segments)

    return processed_text, segments


def tokenize_text(model, text):
    """ Works for token-based and character-based models """
    if hasattr(model, "tokenizer"):
        return model.tokenizer.text_to_ids(text)

    elif hasattr(model.decoder, "vocabulary"):
        # i.e. assume it is character-based model (in theory could be TalkNet - that will not work)
        tokens = []
        for character in text:
            if character in model.decoder.vocabulary:
                tokens.append(model.decoder.vocabulary.index(character))
            else:
                tokens.append(len(model.decoder.vocabulary))  # return unk token (same as blank token)

        return tokens

    else:
        raise RuntimeError("Can't tokenize this model atm")


def get_log_probs_y_T_U(data, model, separator):
    """
    Preparing some tensors to be used during Viterbi decoding.
    Returns:
        log_probs, y, T, U_dash (ie. U with every other token being a blank)
    """

    audio_filepaths = [line["audio_filepath"] for line in data]
    B = len(audio_filepaths)

    with torch.no_grad():
        hypotheses = model.transcribe(audio_filepaths, return_hypotheses=True, batch_size=B)

    log_probs_list = []
    T_list = []
    for hypothesis in hypotheses:
        log_probs_list.append(hypothesis.y_sequence)
        T_list.append(hypothesis.y_sequence.shape[0])

    y_list = []
    U_list = []
    for line in data:
        processed_text, _ = get_processed_text_and_segments(line['text'], separator)
        y_line = tokenize_text(model, processed_text)
        y_list.append(y_line)
        U_list.append(len(y_line))

    T_max = max(T_list)
    U_max = max(U_list)
    blank_index = len(model.decoder.vocabulary)
    V = len(model.decoder.vocabulary) + 1
    T = torch.tensor(T_list)

    # make log_probs tensor of shape (B x T_max x V)
    log_probs = V_NEG_NUM * torch.ones((B, T_max, V))
    for b, log_probs_b in enumerate(log_probs_list):
        t = log_probs_b.shape[0]
        log_probs[b, :t, :] = log_probs_b

    # make y tensor of shape (B x U_dash_max)
    U_dash_max = 2 * U_max + 1
    y = V * torch.ones((B, U_dash_max), dtype=torch.int64)
    U_dash_list = []
    for b, y_b in enumerate(y_list):
        y_dash_b = [blank_index]
        for y_b_element in y_b:
            y_dash_b.append(y_b_element)
            y_dash_b.append(blank_index)
        U_dash = len(y_dash_b)
        y[b, :U_dash] = torch.tensor(y_dash_b)
        U_dash_list.append(U_dash)

    U_dash = torch.tensor(U_dash_list)

    # transfer all tensors to device
    log_probs = log_probs.to(model.device)
    y = y.to(model.device)
    T = T.to(model.device)
    U_dash = U_dash.to(model.device)

    return log_probs, y, T, U_dash
