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
import torch

V_NEG_NUM = -1e30


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


def get_log_probs_y_T_U(data, model):
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
        y_line = tokenize_text(model, line["text"])
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

    return log_probs, y, T, U_dash

def make_basetoken_ctm(
    data, alignments, model, model_downsample_factor, output_ctm_folder, utt_id_extractor_func, audio_sr,
):
    """
    Note: assume order of utts in data matches order of utts in alignments
    """
    assert len(data) == len(alignments)

    os.makedirs(output_ctm_folder, exist_ok=True)

    blank_index = len(model.decoder.vocabulary)
    spectrogram_hop_length = model.preprocessor.featurizer.hop_length
    timestep_to_sample_ratio = spectrogram_hop_length * model_downsample_factor

    for manifest_line, alignment in zip(data, alignments):

        # alignments currently list into loc in sentence.
        # we want to get token_ids for utterance
        token_ids = tokenize_text(model, manifest_line["text"])
        u_list = [blank_index]
        for token_id in token_ids:
            u_list.extend([token_id, blank_index])

        # make 'basetokens_info' - a list of dictionaries:
        # e.g. [
        #   {"basetoken": "<blank>", "u": 0, "t_start": 0, "t_end", 1 },
        #   {"basetoken": "_sh", "u": 0, "t_start": 2, "t_end", 4 },
        #   ...
        # ]
        basetokens_info = []
        u = 0
        for t, u in enumerate(alignment):
            alignment_token_in_vocab = u_list[u]
            if alignment_token_in_vocab == blank_index:
                alignment_token_as_string = "<blank>"
            else:
                alignment_token_as_string = model.decoder.vocabulary[alignment_token_in_vocab]
                if alignment_token_as_string == " ":
                    alignment_token_as_string = "<space>"

            if t == 0:
                basetokens_info.append(
                    {"basetoken": alignment_token_as_string, "u": u, "t_start": t,}
                )

            else:
                if u == basetokens_info[-1]["u"]:
                    pass
                else:
                    basetokens_info[-1]["t_end"] = t - 1
                    basetokens_info.append(
                        {"basetoken": alignment_token_as_string, "u": u, "t_start": t,}
                    )

        basetokens_info[-1]["t_end"] = len(alignment) - 1

        utt_id = utt_id_extractor_func(manifest_line["audio_filepath"])

        with open(os.path.join(output_ctm_folder, f"{utt_id}.ctm"), "w") as f_ctm:
            for basetoken_info in basetokens_info:
                basetoken = basetoken_info["basetoken"]
                start_sample = basetoken_info["t_start"] * timestep_to_sample_ratio
                end_sample = (basetoken_info["t_end"] + 1) * timestep_to_sample_ratio - 1

                start_time = start_sample / audio_sr
                end_time = end_sample / audio_sr

                f_ctm.write(f"{utt_id} 1 {start_time:.5f} {end_time - start_time:.5f} {basetoken}\n")

        return None


def make_word_ctm(
    data, alignments, model, model_downsample_factor, output_ctm_folder, utt_id_extractor_func, audio_sr,
):
    """
    Note: assume order of utts in data matches order of utts in alignments
    """
    assert len(data) == len(alignments)

    os.makedirs(output_ctm_folder, exist_ok=True)

    spectrogram_hop_length = model.preprocessor.featurizer.hop_length
    timestep_to_sample_ratio = spectrogram_hop_length * model_downsample_factor

    for manifest_line, alignment in zip(data, alignments):
        # make 'words_info' - a list of dictionaries:
        # e.g. [
        #   {"word": "h#", "u_start": 0, "u_end": 0, "t_start": 0, "t_end", 0 },
        #   {"word": "she", "u_start": 1, "u_end": 5, "t_start": 1, "t_end", 5 },
        #   {"word": "had", "u_start": 7, "u_end": 10, "t_start": 6, "t_end", 8 }
        #   ...
        # ]
        words_info = [{"word": "h#", "u_start": 0, "u_end": 0, "t_start": 0, "t_end": None}]
        u_counter = 1
        if " " not in model.decoder.vocabulary:
            for word in manifest_line["text"].split(" "):
                word_info = {
                    "word": word,
                    "u_start": u_counter,
                    "u_end": None,
                    "t_start": None,
                    "t_end": None,
                }
                word_tokens = tokenize_text(model, word)
                word_info["u_end"] = word_info["u_start"] + 2 * len(word_tokens) - 2
                u_counter += 2 * len(word_tokens)

                words_info.append(word_info)

        else:

            for word_i, word in enumerate(manifest_line["text"].split(" ")):
                word_info = {
                    "word": word, 
                    "u_start": u_counter,
                    "u_end": None,
                    "t_start": None,
                    "t_end": None,
                }
                word_tokens = tokenize_text(model, word)

                word_info["u_end"] = word_info["u_start"] + 2 * len(word_tokens) - 2
                u_counter += 2 * len(word_tokens)

                words_info.append(word_info)

                if word_i < len(manifest_line["text"].split(" ")) - 1:
                    # add the space after every word except the final word
                    word_info = {
                        "word": "<space>", 
                        "u_start": u_counter,
                        "u_end": None,
                        "t_start": None,
                        "t_end": None,
                    }
                    word_tokens = tokenize_text(model, " ")

                    word_info["u_end"] = word_info["u_start"] + 2 * len(word_tokens) - 1
                    u_counter += 2 * len(word_tokens)

                    words_info.append(word_info)

        words_info_pointer = 0
        for t, u_at_t in enumerate(alignment):
            if words_info_pointer < len(words_info):
                if u_at_t == words_info[words_info_pointer]["u_start"]:
                    if words_info[words_info_pointer]["t_start"] is None:
                        words_info[words_info_pointer]["t_start"] = t

                if u_at_t > words_info[words_info_pointer]["u_end"]:
                    if words_info[words_info_pointer]["t_end"] is None:
                        words_info[words_info_pointer]["t_end"] = t - 1

                        words_info_pointer += 1

                if words_info_pointer < len(words_info):
                    if u_at_t == words_info[words_info_pointer]["u_start"]:
                        if words_info[words_info_pointer]["t_start"] is None:
                            words_info[words_info_pointer]["t_start"] = t

        # don't forget the final t_end as our for-loop might miss it
        if alignment[-1] == words_info[-1]["u_end"]:
            words_info[-1]["t_end"] = len(alignment) - 1

        utt_id = utt_id_extractor_func(manifest_line["audio_filepath"])

        with open(os.path.join(output_ctm_folder, f"{utt_id}.ctm"), "w") as f_ctm:
            for word_info in words_info:
                if not (word_info["word"] == "initial_silence" and word_info["t_end"] == None):
                    word = word_info["word"]
                    # print('word_info', word_info)
                    start_sample = word_info["t_start"] * timestep_to_sample_ratio
                    end_sample = (word_info["t_end"] + 1) * timestep_to_sample_ratio - 1

                    start_time = start_sample / audio_sr
                    end_time = end_sample / audio_sr

                    f_ctm.write(f"{utt_id} 1 {start_time:.5f} {end_time - start_time:.5f} {word}\n")

    return None
