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

from .data_prep import get_processed_text_and_segments, tokenize_text


def _get_utt_id(audio_filepath, n_parts_for_ctm_id):
    fp_parts = Path(audio_filepath).parts[-n_parts_for_ctm_id:]
    utt_id = Path("_".join(fp_parts)).stem
    utt_id = utt_id.replace(" ", "-")  # replace any spaces in the filepath with dashes
    return utt_id


def make_basetoken_ctm(
    data, alignments, model, model_downsample_factor, output_ctm_folder, n_parts_for_ctm_id, audio_sr,
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

        # get utt_id that will be used for saving CTM file as <utt_id>.ctm
        utt_id = _get_utt_id(manifest_line['audio_filepath'], n_parts_for_ctm_id)

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
    data, alignments, model, model_downsample_factor, output_ctm_folder, n_parts_for_ctm_id, audio_sr, separator
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
        #   {"word": "<initial_silence>", "u_start": 0, "u_end": 0, "t_start": 0, "t_end", 0 },
        #   {"word": "she", "u_start": 1, "u_end": 5, "t_start": 1, "t_end", 5 },
        #   {"word": "had", "u_start": 7, "u_end": 10, "t_start": 6, "t_end", 8 }
        #   ...
        # ]
        words_info = [{"word": "<initial_silence>", "u_start": 0, "u_end": 0, "t_start": 0, "t_end": None}]
        u_counter = 1
        _, segments = get_processed_text_and_segments(manifest_line['text'], separator)
        if separator not in model.decoder.vocabulary:
            for word in segments:
                word_info = {
                    "word": word.replace(" ", "<space>"),
                    "u_start": u_counter,
                    "u_end": None,
                    "t_start": None,
                    "t_end": None,
                }
                word_tokens = tokenize_text(model, word)
                word_info["u_end"] = word_info["u_start"] + 2 * (len(word_tokens) - 1)
                words_info.append(word_info)

                u_counter += 2 * len(word_tokens)

                if " " in model.decoder.vocabulary:
                    u_counter += 2

        else:
            for word_i, word in enumerate(segments):
                word_info = {
                    "word": word.replace(" ", "<space>"),
                    "u_start": u_counter,
                    "u_end": None,
                    "t_start": None,
                    "t_end": None,
                }
                word_tokens = tokenize_text(model, word)

                word_info["u_end"] = word_info["u_start"] + 2 * (len(word_tokens) - 1)
                u_counter += 2 * len(word_tokens)

                words_info.append(word_info)

                if word_i < len(manifest_line["text"].split(separator)) - 1:
                    # add the space after every word except the final word
                    word_info = {
                        "word": separator.replace(" ", "<space>"),
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

        # get utt_id that will be used for saving CTM file as <utt_id>.ctm
        utt_id = _get_utt_id(manifest_line['audio_filepath'], n_parts_for_ctm_id)

        with open(os.path.join(output_ctm_folder, f"{utt_id}.ctm"), "w") as f_ctm:
            for word_info in words_info:
                if not (word_info["word"] == "initial_silence" and word_info["t_end"] is None):
                    word = word_info["word"]
                    start_sample = word_info["t_start"] * timestep_to_sample_ratio
                    end_sample = (word_info["t_end"] + 1) * timestep_to_sample_ratio - 1

                    start_time = start_sample / audio_sr
                    end_time = end_sample / audio_sr

                    f_ctm.write(f"{utt_id} 1 {start_time:.5f} {end_time - start_time:.5f} {word}\n")

    return None
