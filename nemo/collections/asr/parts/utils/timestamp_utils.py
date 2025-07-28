# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import re
from typing import List, Optional, Set, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from nemo.collections.asr.parts.submodules.ctc_decoding import AbstractCTCDecoding
from nemo.collections.asr.parts.utils.aligner_utils import (
    BLANK_TOKEN,
    Segment,
    Word,
    add_t_start_end_to_utt_obj,
    get_batch_variables,
    viterbi_decoding,
)
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


def process_aed_timestamp_outputs(outputs, subsampling_factor: int = 1, window_stride: float = 0.01):
    """
    Processes AED timestamp outputs and extracts word-level timestamps.
    Args:
        outputs (Hypothesis, list of Hypotesis or list of list of Hypotesis): The hypothesis outputs to process. Can be a single Hypothesis object or a list of Hypothesis objects.
        subsampling_factor (int, optional): The subsampling factor used in the model. Default is 1.
        window_stride (float, optional): The window stride used in the model. Default is 0.01.
    Returns:
        list of list of Hypotesis: The processed hypothesis outputs with word-level timestamps added.
    """

    def extract_words_with_timestamps(text, subsampling_factor: int = 1, window_stride: float = 0.01):
        text = text.strip()  # remove leading and trailing whitespaces - training data artifact

        if not re.search(r'<\|\d+\|>.*?<\|\d+\|>', text):
            return None, text

        # Find words that directly have start and end timestamps
        pattern = r'<\|(\d+)\|>(.*?)<\|(\d+)\|>'

        matches = []
        text_without_timestamps = []
        for match in re.finditer(pattern, text):
            start_offset = int(match.group(1))
            content = match.group(2)
            end_offset = int(match.group(3))
            start_time = start_offset * window_stride * subsampling_factor
            end_time = end_offset * window_stride * subsampling_factor

            # Only include if there's actual content
            if content.strip():
                sample = {
                    'word': content.strip(),
                    'start_offset': start_offset,
                    'end_offset': end_offset,
                    'start': start_time,
                    'end': end_time,
                }
                matches.append(sample)
                text_without_timestamps.append(content.strip())

        text_without_timestamps = ' '.join(text_without_timestamps)
        return matches, text_without_timestamps

    def segments_offset_to_time(segments, window_stride, subsampling_factor):
        for segment in segments:
            segment['start'] = segment['start_offset'] * window_stride * subsampling_factor
            segment['end'] = segment['end_offset'] * window_stride * subsampling_factor
        return segments

    def process_hypothesis(hyp, subsampling_factor: int, window_stride: float):
        """
        Processes a single Hypothesis object to extract timestamps.
        """
        timestamp, text = extract_words_with_timestamps(hyp.text, subsampling_factor, window_stride)
        hyp.text = text
        if timestamp is not None:
            if len(hyp.timestamp) == 0:
                hyp.timestamp = {}

            hyp.timestamp.update(
                {
                    'word': timestamp,
                    'segment': [],
                    'char': [],  # not supported for AED
                }
            )

            segments = AbstractCTCDecoding._get_segment_offsets(timestamp, segment_delimiter_tokens=['.', '?', '!'])
            hyp.timestamp['segment'] = segments_offset_to_time(segments, window_stride, subsampling_factor)
        else:
            hyp.timestamp = {
                'word': [],
                'segment': [],
                'char': [],
            }

        return hyp

    if outputs is None:
        return outputs

    if isinstance(outputs, Hypothesis):
        return [process_hypothesis(outputs, subsampling_factor, window_stride)]
    elif isinstance(outputs, list) and isinstance(outputs[0], Hypothesis):
        # list of Hypothesis
        return [process_hypothesis(hyp, subsampling_factor, window_stride) for hyp in outputs]
    elif isinstance(outputs, list) and isinstance(outputs[0], list) and isinstance(outputs[0][0], Hypothesis):
        # list of list of Hypothesis (for beam decoding)
        return [
            [process_hypothesis(hyp, subsampling_factor, window_stride) for hyp in hyps_list] for hyps_list in outputs
        ]
    else:
        raise ValueError(
            f"Expected Hypothesis, list of Hypothesis or list of list of Hypothesis object, got {type(outputs)}"
        )


def process_timestamp_outputs(outputs, subsampling_factor: int = 1, window_stride: float = 0.01):
    """
    Process the timestamps from list of hypothesis to user friendly format.
    Converts the start and end duration from frames to seconds.
    Args:
        outputs: List of Hypothesis objects.
        subsampling_factor: int, Subsampling factor used in the model.
        window_stride: float, Window stride used in the model. (sometimes referred to as hop length/shift)
    Returns:
        List of Hypothesis objects with processed timestamps
    """

    if outputs is None:
        return outputs

    if isinstance(outputs, Hypothesis):
        outputs = [outputs]

    if not isinstance(outputs[0], Hypothesis):
        raise ValueError(f"Expected Hypothesis object, got {type(outputs[0])}")

    def process_timestamp(timestamp, subsampling_factor, window_stride):
        """
        Process the timestamp for a single hypothesis.
        return the start and end duration in seconds.
        """
        for idx, val in enumerate(timestamp):
            start_offset = val['start_offset']
            end_offset = val['end_offset']
            start = start_offset * window_stride * subsampling_factor
            end = end_offset * window_stride * subsampling_factor
            val['start'] = start
            val['end'] = end

        return timestamp

    for idx, hyp in enumerate(outputs):
        if not hasattr(hyp, 'timestamp'):
            raise ValueError(
                f"Expected Hypothesis object to have 'timestamp' attribute, when compute_timestamps is \
                    enabled but got {hyp}"
            )
        timestamp = hyp.timestamp
        if 'word' in timestamp:
            outputs[idx].timestamp['word'] = process_timestamp(timestamp['word'], subsampling_factor, window_stride)
        if 'char' in timestamp:
            outputs[idx].timestamp['char'] = process_timestamp(timestamp['char'], subsampling_factor, window_stride)
        if 'segment' in timestamp:
            outputs[idx].timestamp['segment'] = process_timestamp(
                timestamp['segment'], subsampling_factor, window_stride
            )
    return outputs


def get_forced_aligned_timestamps_with_external_model(
    audio: Union[str, List[str], np.ndarray, DataLoader],
    external_ctc_model,
    main_model_predictions: List[Hypothesis],
    batch_size: int = 4,
    viterbi_device: Optional[torch.device] = None,
    supported_punctuation: Optional[Union[Set, List[str]]] = {',', '.', '!', '?'},
):

    def process_timestamps(utt_obj, output_timestep_duration):
        timestamps = {
            "segment": [],
            "word": [],
            "char": [],
        }

        for segment in utt_obj.segments_and_tokens:

            if not isinstance(segment, Segment):
                continue

            timestamps["segment"].append(
                {
                    "segment": segment.text,
                    "start_offset": int(segment.t_start / output_timestep_duration) if segment.t_start != -1 else -1,
                    "end_offset": int(segment.t_end / output_timestep_duration) if segment.t_end != -1 else -1,
                    "start": round(segment.t_start, 2),
                    "end": round(segment.t_end, 2),
                }
            )

            for word in segment.words_and_tokens:

                if not isinstance(word, Word):
                    continue

                timestamps["word"].append(
                    {
                        "word": word.text,
                        "start_offset": int(word.t_start / output_timestep_duration) if word.t_start != -1 else -1,
                        "end_offset": int(word.t_end / output_timestep_duration) if word.t_end != -1 else -1,
                        "start": round(word.t_start, 2),
                        "end": round(word.t_end, 2),
                    }
                )

                for idx, token in enumerate(word.tokens):
                    if token.text == BLANK_TOKEN:
                        continue

                    if token.text in supported_punctuation:

                        previous_token_end = (
                            round(timestamps['char'][-1]['end'], 2) if timestamps['char'] else round(token.t_start, 2)
                        )

                        if segment.t_end == word.t_end:
                            timestamps["segment"][-1]["end"] = previous_token_end
                            timestamps["segment"][-1]["end_offset"] = int(previous_token_end / output_timestep_duration) if previous_token_end != -1 else -1
                        if word.t_end == token.t_end:
                            timestamps["word"][-1]["end"] = previous_token_end
                            timestamps["word"][-1]["end_offset"] = int(previous_token_end / output_timestep_duration) if previous_token_end != -1 else -1


                        token.t_end = token.t_start = previous_token_end

                    timestamps["char"].append(
                        {
                            "char": token.text,
                            "start_offset": (
                                int(token.t_start / output_timestep_duration) if token.t_start != -1 else -1
                            ),
                            "end_offset": int(token.t_end / output_timestep_duration) if token.t_end != -1 else -1,
                            "start": round(token.t_start, 2),
                            "end": round(token.t_end, 2),
                        }
                    )

        return timestamps

    if viterbi_device is None:
        viterbi_device = external_ctc_model.device

    for start_idx in range(0, len(audio), batch_size):
        end_idx = start_idx + batch_size

        audio_batch = [audio[i] for i in range(start_idx, end_idx)]

        log_probs_batch, y_batch, T_batch, U_batch, utt_obj_batch, output_timestep_duration = get_batch_variables(
            audio=audio_batch,
            model=external_ctc_model,
            separator=['.', '?', '!'],
            gt_text_batch=[hyp.text for hyp in main_model_predictions[start_idx:end_idx]],
        )

        alignments_batch = viterbi_decoding(
            log_probs_batch,
            y_batch,
            T_batch,
            U_batch,
            viterbi_device=viterbi_device,
        )

        for i, (utt_obj, alignment_utt) in enumerate(zip(utt_obj_batch, alignments_batch)):
            utt_obj = add_t_start_end_to_utt_obj(utt_obj, alignment_utt, output_timestep_duration)
            main_model_predictions[start_idx + i].timestamp = process_timestamps(utt_obj, output_timestep_duration)

    return main_model_predictions