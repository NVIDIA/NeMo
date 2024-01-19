# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#

from typing import List, Union
from nemo.collections.asr.parts.utils import rnnt_utils
from ctc_based_word_spotter import WSHyp
import numpy as np


def merge_alignment_with_ws_hyps(
    candidate: Union[np.ndarray, rnnt_utils.Hypothesis],
    asr_model,
    cb_results: List[WSHyp],
    decoder_type: str = "ctc",
    intersection_threshold: float = 30.0,
    blank_idx: int = 0,
) -> str:
    """
    Merge context biasing predictions with ctc/rnnt word-level alignment.
    Words from alignment will be replaced by spotted words if intersection between them is greater than threshold.

    Args:
        candidate: argmax predictions per frame (for ctc) or rnnt hypothesis (for rnnt)
        asr_model: ctc or hybrid transducer-ctc model
        cb_results: list of context biasing predictions (spotted words)
        decoder_type: ctc or rnnt
        intersection_threshold: threshold for intersection between spotted word and word from alignment (in percentage)
    Returns:
        boosted_text: final text with context biasing predictions
    """

    # step 1: get token-level alignment [frame, token]
    if decoder_type == "ctc":
        alignment_tokens = []
        prev_token = None
        for idx, token in enumerate(candidate):
            if token != blank_idx:
                if token == prev_token:
                    alignment_tokens[-1] = [idx, asr_model.tokenizer.ids_to_tokens([int(token)])[0]]
                else:
                    alignment_tokens.append([idx, asr_model.tokenizer.ids_to_tokens([int(token)])[0]])
            prev_token = token
        
    elif decoder_type == "rnnt":
        alignment_tokens = []
        tokens = asr_model.tokenizer.ids_to_tokens(candidate.y_sequence.tolist())
        for idx, token in enumerate(tokens):
            alignment_tokens.append([candidate.timestep[idx], token])

    if not alignment_tokens:
        # ctc/rnnt decoding results are empty, return context biasing results only
        return " ".join([ws_hyp.word for ws_hyp in cb_results])

    # step 2: get word-level alignment [word, start_frame, end_frame]
    bow = "▁" # symbol for begin of word (bow) in BPE tokenizer
    word_alignment = []
    word = ""
    l, r, = None, None
    for item in alignment_tokens:
        if not word:
            word = item[1][1:]
            l = r = item[0]
        else:
            if item[1].startswith(bow):
                word_alignment.append((word, l, r))
                word = item[1][1:]
                l = r = item[0]
            else:
                word += item[1]
                r = item[0]
    word_alignment.append((word, l, r))
    ref_text = [item[0] for item in word_alignment]
    ref_text = " ".join(ref_text)

    # step 3: merge spotted words with word alignment
    for ws_hyp in cb_results:
        # extend ws_hyp start frame in case of rnnt (rnnt tends to predict labels one frame earlier sometimes)
        if ws_hyp.start_frame > 0 and decoder_type == "rnnt":
            ws_hyp.start_frame -= 1
        new_word_alignment = []
        already_inserted = False
        # get interval of spotted word
        ws_interval = set(range(ws_hyp.start_frame, ws_hyp.end_frame+1))
        for item in word_alignment:
            # get interval if word from alignment
            li, ri = item[1], item[2]
            item_interval = set(range(li, ri+1))
            if ws_hyp.start_frame < li:
                # spotted word starts before first word from alignment
                if not already_inserted:
                    new_word_alignment.append((ws_hyp.word, ws_hyp.start_frame, ws_hyp.end_frame))
                    already_inserted = True
            # compute intersection between spotted word and word from alignment in percentage
            intersection_part = 100/len(item_interval) * len(ws_interval & item_interval)
            if intersection_part <= intersection_threshold:
                new_word_alignment.append(item)
            elif not already_inserted:
                # word from alignment will be replaced by spotted word
                new_word_alignment.append((ws_hyp.word, ws_hyp.start_frame, ws_hyp.end_frame))
                already_inserted = True
        # insert last spotted word if not yet
        if not already_inserted:
            new_word_alignment.append((ws_hyp.word, ws_hyp.start_frame, ws_hyp.end_frame))

        word_alignment = new_word_alignment

    boosted_text_list = [item[0] for item in new_word_alignment]
    boosted_text = " ".join(boosted_text_list)
    
    return boosted_text