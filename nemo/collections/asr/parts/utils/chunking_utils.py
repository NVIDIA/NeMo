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

from nemo.collections.asr.parts.utils.timestamp_utils import (
    get_segment_offsets,
    get_words_offsets,
)
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
import torch

def merge_parallel_chunks(
    hypotheses, 
    encoded_len, 
    model, 
    subsampling_factor, 
    window_stride,
    tokenizer
):
    """
    Merges hypotheses from parallel chunks into a single hypothesis with proper text, 
    token sequences, and timestamps.
    
    Args:
        hypotheses: List of Hypothesis objects from each chunk
        encoded_len: Tensor of encoded lengths for each chunk
        model: The ASR model instance (needed for LCS alignment)
        subsampling_factor: The encoder's subsampling factor
        window_stride: The preprocessor's window stride
        tokenizer: The tokenizer instance for converting tokens to text
        
    Returns:
        Hypothesis: A single merged hypothesis with combined text, tokens, and timestamps
    """
    
    # we take the overlap to be 1 second, and count number of tokens in it
    delay = int(1 / (subsampling_factor / 100))
    
    merged_tokens = hypotheses[0].y_sequence.tolist()
    #avoid circular import
    from nemo.collections.asr.parts.utils.streaming_utils import lcs_alignment_merge_buffer
    for i in range(1, len(hypotheses)):
        merged_tokens = lcs_alignment_merge_buffer(
            buffer=merged_tokens,
            data=hypotheses[i].y_sequence.tolist()[
                : int(delay * 0.6)
            ],  # only approximately 60% of the tokens are non blank
            delay=delay,
            model=model,
            max_steps_per_timestep=1,
            min_lcs_length=1,
            parallel_chunking=True,
        )
        merged_tokens += hypotheses[i].y_sequence.tolist()[int(delay * 0.6) :]
    
    # Convert merged tokens to text
    final_text = tokenizer.ids_to_text(merged_tokens)
    
    merged_hypotheses = Hypothesis(
        score=0.0,
        y_sequence=torch.tensor([]),
        timestamp={
            'word': [],
            'segment': [],
        },
    )
    
    chunk_offsets = [0] + [x * subsampling_factor for x in encoded_len.tolist()]
    
    merged_hypotheses = join_y_sequence(merged_hypotheses, hypotheses)
    merged_hypotheses.text = final_text
    
    # Merge timestamps and add word and segment level timestamps
    merged_hypotheses = join_timestamp_and_add_word_and_segment_level_timestamps(
        merged_hypotheses, hypotheses, chunk_offsets, subsampling_factor, window_stride, merged_tokens
    )
    
    return merged_hypotheses

def join_y_sequence(merged_hypothesis, hypotheses):
        merged_hypothesis.y_sequence = torch.cat([h.y_sequence for h in hypotheses])
        return merged_hypothesis


def join_timestamp_and_add_word_and_segment_level_timestamps(
        merged_hypotheses, hypotheses, chunk_offsets, subsampling_factor, window_stride, merged_tokens=None
    ):
    # Initialize empty timestamp structure

    # First, combine char-level timestamps from all chunks
    char_timestamps = join_char_level_timestamps(hypotheses, chunk_offsets, subsampling_factor, window_stride, merged_tokens)

    # Create encoded_char_offsets for word/segment generation
    encoded_char_offsets = []
    for char_offset in char_timestamps:
        enc_char_offset = char_offset.copy()
        enc_char_offset['char'] = enc_char_offset['token']
        encoded_char_offsets.append(enc_char_offset)

    # Generate word-level timestamps from combined char timestamps
    word_offsets = get_words_offsets(
        char_offsets=char_timestamps,
        encoded_char_offsets=encoded_char_offsets,
        supported_punctuation={',', '.', '!', '?'},
    )

    # Generate segment-level timestamps from word timestamps
    segment_offsets = get_segment_offsets(
        word_offsets=word_offsets, segment_delimiter_tokens={'.', '!', '?', "..."}
    )
    # Update the merged hypothesis with word and segment timestamps
    merged_hypotheses.timestamp['word'] = word_offsets
    merged_hypotheses.timestamp['segment'] = segment_offsets

    return merged_hypotheses

def join_char_level_timestamps(
    hypotheses,
    chunk_offsets,
    subsampling_factor,
    window_stride,
    merged_tokens=None,
):
    char_timestamps = []
    cumulative_offset = 0  # raw (pre-subsampling) frames already emitted
    overall_removed_offset = 0  # subsampled frames trimmed so far
    j_token = 0  # cursor in merged_tokens

    subsamp = subsampling_factor
    stride = window_stride # sec per raw frame

    for i, h in enumerate(hypotheses):
        cumulative_offset += chunk_offsets[i]            # raw frames
        chunk_frame_offset = cumulative_offset // subsamp

        # 1) figure out how much of the *front* of this chunk we will drop
        removed_in_chunk = 0
        for char in h.timestamp['char']:
            keep = (
                merged_tokens is None
                or (j_token < len(merged_tokens)
                    and char and char['token_id'] == merged_tokens[j_token])
            )
            if keep:                            
                break
            if char and char['end_offset'] != -1:
                removed_in_chunk = char['end_offset'] + 1

        for char in h.timestamp['char']:
            if not char:
                continue
            keep = (
                merged_tokens is None
                or (j_token < len(merged_tokens)
                    and char['token_id'] == merged_tokens[j_token])
            )
            if not keep:
                continue

            # adjust offsets: chunk start + global chunk shift − total removed
            start_off = char['start_offset']
            end_off   = char['end_offset']


            upd = dict(char)
            if start_off != -1:
                upd['start_offset'] = (
                    start_off
                    + chunk_frame_offset  # place chunk globally
                    - overall_removed_offset  # past trims
                    - removed_in_chunk  # trims in this chunk
                )
            if end_off != -1:
                upd['end_offset'] = (
                    end_off
                    + chunk_frame_offset
                    - overall_removed_offset
                    - removed_in_chunk
                )

            # convert to seconds
            upd['start'] = (
                -1 if upd['start_offset'] == -1
                else upd['start_offset'] * stride * subsamp
            )
            upd['end'] = (
                -1 if upd['end_offset'] == -1
                else upd['end_offset'] * stride * subsamp
            )

            char_timestamps.append(upd)
            j_token += 1

        # 3) make the trim visible to later chunks
        overall_removed_offset += removed_in_chunk

    return char_timestamps
