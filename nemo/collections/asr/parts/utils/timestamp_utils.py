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
from typing import Callable, Dict, List, Optional, Set, Union

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
from nemo.utils import logging, logging_mode


def get_words_offsets(
        char_offsets: List[Dict[str, Union[str, float]]],
        encoded_char_offsets: List[Dict[str, Union[str, float]]],
        word_delimiter_char: str = " ",
        tokenizer_type: str = "bpe",
        supported_punctuation: Optional[Set] = None,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Utility method which constructs word time stamps out of sub-word time stamps.

        **Note**: Only supports Sentencepiece based tokenizers !

        Args:
            char_offsets: A list of dictionaries, each containing "char", "start_offset" and "end_offset",
                        where "char" is decoded with the tokenizer.
            encoded_char_offsets: A list of dictionaries, each containing "char", "start_offset" and "end_offset",
                        where "char" is the original id/ids from the hypotheses (not decoded with the tokenizer).
                        This is needed for subword tokenization models.
            word_delimiter_char: Character token that represents the word delimiter. By default, " ".
            supported_punctuation: Set containing punctuation marks in the vocabulary.

        Returns:
            A list of dictionaries containing the word offsets. Each item contains "word", "start_offset" and
            "end_offset".
        """

        def define_word_start_condition() -> Callable[[str, str], bool]:
            """
            Define the word start condition based on the tokenizer type and word delimiter character.
            """
            if word_delimiter_char == " ":
                if tokenizer_type == "wpe":
                    return lambda token, token_text: token_text and not token_text.startswith("##")
                return lambda token, token_text: token != token_text
            else:
                return lambda token, token_text: token_text == word_delimiter_char
            

        word_offsets = []
        previous_token_index = 0

        # Built tokens should be list here as when dealing with wpe tokenizer,
        # ids should be decoded together to ensure tokens starting with ## are not split
        built_word = ""

        condition_for_word_start = define_word_start_condition()

        # For every collapsed sub-word token
        for i, (char_offset, char_token_offset) in enumerate(zip(char_offsets, encoded_char_offsets)):

            char_text = char_offset['char'].strip()
            char_token = char_token_offset['char']

            curr_punctuation = supported_punctuation and char_text in supported_punctuation

            # It is a sub-word token, or contains an identifier at the beginning such as _ or ## that was stripped
            # after forcing partial text conversion of the token.
            # AND it is not a supported punctuation mark, which needs to be added to the built word regardless of its identifier.
            if condition_for_word_start(char_token, char_text) and not curr_punctuation:
                # If there are any partially or fully built sub-word token ids, construct to text.
                # Note: This is "old" subword, that occurs *after* current sub-word has started.
                if built_word:
                    word_offsets.append(
                        {
                            "word": built_word,
                            "start_offset": char_offsets[previous_token_index]["start_offset"],
                            "end_offset": char_offsets[i - 1]["end_offset"],
                        }
                    )

                    if "start" in char_offset:
                        word_offsets[-1]["start"] = char_offsets[previous_token_index]["start"]
                    if "end" in char_offset:
                        word_offsets[-1]["end"] = char_offsets[i - 1]["end"]

                # Prepare new built_word
                built_word = ""

                if char_text != word_delimiter_char:
                    built_word = char_text
                    previous_token_index = i

            # If the token is a punctuation mark and there is no built word, then the previous word is complete
            # and lacks the punctuation mark. We need to add the punctuation mark to the previous formed word.
            elif curr_punctuation and not built_word:
                last_built_word = word_offsets[-1]
                last_built_word['end_offset'] = char_offset['end_offset']
                if last_built_word['word'][-1] == ' ':
                    last_built_word['word'] = last_built_word['word'][:-1]
                last_built_word['word'] += char_text
            else:
                # If the token does not contain any sub-word start mark, then the sub-word has not completed yet
                # Append to current built word.
                # If this token is the first in the built_tokens, we should save its index as the previous token index
                # because it will be used to calculate the start offset of the word.
                if not built_word:
                    previous_token_index = i
                built_word += char_text

        # Inject the start offset of the first token to word offsets
        # This is because we always skip the delay the injection of the first sub-word due to the loop
        # condition and check whether built token is ready or not.
        # Therefore without this forced injection, the start_offset appears as off by 1.
        if len(word_offsets) == 0:
            # alaptev: sometimes word_offsets can be empty
            if built_word:
                word_offsets.append(
                    {
                        "word": built_word,
                        "start_offset": char_offsets[0]["start_offset"],
                        "end_offset": char_offsets[-1]["end_offset"],
                    }
                )
                if "start" in char_offsets[0]:
                    word_offsets[0]["start"] = char_offsets[0]["start"]
                if "end" in char_offsets[-1]:
                    word_offsets[-1]["end"] = char_offsets[-1]["end"]
        else:
            word_offsets[0]["start_offset"] = char_offsets[0]["start_offset"]

            if "start" in char_offsets[0]:
                word_offsets[0]["start"] = char_offsets[0]["start"]

            # If there are any remaining tokens left, inject them all into the final word offset.
            # Note: The start offset of this token is the start time of the first token inside build_token.
            # Note: The end offset of this token is the end time of the last token inside build_token
            if built_word:
                word_offsets.append(
                    {
                        "word": built_word,
                        "start_offset": char_offsets[previous_token_index]["start_offset"],
                        "end_offset": char_offsets[-1]["end_offset"],
                    }
                )
                if "start" in char_offset:
                    word_offsets[-1]["start"] = char_offsets[previous_token_index]["start"]
                if "end" in char_offset:
                    word_offsets[-1]["end"] = char_offsets[-1]["end"]

        return word_offsets


def get_segment_offsets(
        word_offsets: List[Dict[str, Union[str, float]]],
        segment_delimiter_tokens: List[str],
        supported_punctuation: Optional[Set] = None,
        segment_gap_threshold: Optional[int] = None,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Utility method which constructs segment time stamps out of word time stamps.

        Args:
            offsets: A list of dictionaries, each containing "word", "start_offset" and "end_offset".
            segments_delimiter_tokens: List containing tokens representing the seperator(s) between segments.
            supported_punctuation: Set containing punctuation marks in the vocabulary.
            segment_gap_threshold: Number of frames between 2 consecutive words necessary to form segments out of plain text.

        Returns:
            A list of dictionaries containing the segment offsets. Each item contains "segment", "start_offset" and
            "end_offset".
        """
        if (
            supported_punctuation
            and not set(segment_delimiter_tokens).intersection(supported_punctuation)
            and not segment_gap_threshold
        ):
            logging.warning(
                f"Specified segment seperators are not in supported punctuation {supported_punctuation}. "
                "If the seperators are not punctuation marks, ignore this warning. "
                "Otherwise, specify 'segment_gap_threshold' parameter in decoding config to form segments.",
                mode=logging_mode.ONCE,
            )

        segment_offsets = []
        segment_words = []
        previous_word_index = 0

        # For every offset word
        for i, offset in enumerate(word_offsets):

            word = offset['word']
            if segment_gap_threshold and segment_words:
                gap_between_words = offset['start_offset'] - word_offsets[i - 1]['end_offset']

                if gap_between_words >= segment_gap_threshold:
                    segment_offsets.append(
                        {
                            "segment": ' '.join(segment_words),
                            "start_offset": word_offsets[previous_word_index]["start_offset"],
                            "end_offset": word_offsets[i - 1]["end_offset"],
                        }
                    )

                    if "start" in word_offsets[previous_word_index]:
                        segment_offsets[-1]["start"] = word_offsets[previous_word_index]["start"]
                    if "end" in word_offsets[i - 1]:
                        segment_offsets[-1]["end"] = word_offsets[i - 1]["end"]

                    segment_words = [word]
                    previous_word_index = i
                    continue

            # check if the word ends with any delimeter token or the word itself is a delimeter
            elif word and (word[-1] in segment_delimiter_tokens or word in segment_delimiter_tokens):
                segment_words.append(word)
                if segment_words:
                    segment_offsets.append(
                        {
                            "segment": ' '.join(segment_words),
                            "start_offset": word_offsets[previous_word_index]["start_offset"],
                            "end_offset": offset["end_offset"],
                        }
                    )

                    if "start" in word_offsets[previous_word_index]:
                        segment_offsets[-1]["start"] = word_offsets[previous_word_index]["start"]
                    if "end" in offset:
                        segment_offsets[-1]["end"] = offset["end"]

                segment_words = []
                previous_word_index = i + 1
                continue

            segment_words.append(word)

        if segment_words:
            start_offset = word_offsets[previous_word_index]["start_offset"]
            segment_offsets.append(
                {
                    "segment": ' '.join(segment_words),
                    "start_offset": start_offset,
                    "end_offset": word_offsets[-1]["end_offset"],
                }
            )

            if "start" in word_offsets[previous_word_index]:
                segment_offsets[-1]["start"] = word_offsets[previous_word_index]["start"]
            if "end" in word_offsets[-1]:
                segment_offsets[-1]["end"] = word_offsets[-1]["end"]

        segment_words.clear()

        return segment_offsets


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
    segment_separators: Optional[Union[str, List[str]]] = ['.', '?', '!', '...'],
    word_separator: Optional[str] = " ",
    supported_punctuation: Optional[Union[Set, List[str]]] = {',', '.', '!', '?'},
    timestamp_type: Optional[Union[str, List[str]]] = "all",
):

    def process_timestamps(utt_obj, output_timestep_duration, timestamp_type):
        if isinstance(timestamp_type, str):
            assert timestamp_type in ["segment", "word", "char", "all"], "Invalid timestamp type must be one of: segment, word, char, all"
            timestamp_type = [timestamp_type] if timestamp_type != "all" else ["segment", "word", "char"]
        elif isinstance(timestamp_type, list):
            assert all(t in ["segment", "word", "char", "all"] for t in timestamp_type), "Invalid timestamp type must be one of: segment, word, char, all"
        else:
            raise ValueError("Invalid timestamp type must be one of: segment, word, char, all")

        timestamps = {t: [] for t in timestamp_type}

        for segment in utt_obj.segments_and_tokens:

            if not isinstance(segment, Segment):
                continue

            if "segment" in timestamp_type:
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

                if "word" in timestamp_type:
                    timestamps["word"].append(
                        {
                            "word": word.text,
                            "start_offset": int(word.t_start / output_timestep_duration) if word.t_start != -1 else -1,
                            "end_offset": int(word.t_end / output_timestep_duration) if word.t_end != -1 else -1,
                            "start": round(word.t_start, 2),
                            "end": round(word.t_end, 2),
                        }
                    )

                for i, token in enumerate(word.tokens):
                    if token.text == BLANK_TOKEN:
                        continue

                    if token.text in supported_punctuation:

                        previous_non_blank_token_idx = i-1
                        previous_non_blank_token = word.tokens[previous_non_blank_token_idx] if previous_non_blank_token_idx >= 0 else None

                        while previous_non_blank_token is not None and previous_non_blank_token.text == BLANK_TOKEN:
                            previous_non_blank_token_idx -= 1
                            previous_non_blank_token = word.tokens[previous_non_blank_token_idx] if previous_non_blank_token_idx >= 0 else None

                        previous_token_end = (
                            round(previous_non_blank_token.t_end, 2) if previous_non_blank_token is not None else round(token.t_start, 2)
                        )

                        if "segment" in timestamp_type:
                            if segment.t_end == word.t_end:
                                timestamps["segment"][-1]["end"] = previous_token_end
                                timestamps["segment"][-1]["end_offset"] = (
                                    int(previous_token_end / output_timestep_duration) if previous_token_end != -1 else -1
                            )
                                
                        if "word" in timestamp_type:
                            if word.t_end == token.t_end:
                                timestamps["word"][-1]["end"] = previous_token_end
                                timestamps["word"][-1]["end_offset"] = (
                                    int(previous_token_end / output_timestep_duration) if previous_token_end != -1 else -1
                                )

                        token.t_end = token.t_start = previous_token_end

                    if "char" in timestamp_type:
                        timestamps["char"].append(
                            {
                            "char": external_ctc_model.tokenizer.ids_to_text([token.token_id]),
                            "token_id": token.token_id,
                            "token": token.text,
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

    if timestamp_type == "char":
        segment_separators = None
        word_separator = None
    elif timestamp_type == "word":
        segment_separators = None

    for start_idx in range(0, len(audio), batch_size):
        end_idx = start_idx + batch_size

        audio_batch = [audio[i] for i in range(start_idx, end_idx)]

        log_probs_batch, y_batch, T_batch, U_batch, utt_obj_batch, output_timestep_duration = get_batch_variables(
            audio=audio_batch,
            model=external_ctc_model,
            segment_separators=segment_separators,
            word_separator=word_separator,
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
            main_model_predictions[start_idx + i].timestamp = process_timestamps(
                utt_obj, output_timestep_duration, timestamp_type
            )

    return main_model_predictions