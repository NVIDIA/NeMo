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

from typing import List, Optional, Dict, Union
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.context_biasing.ctc_based_word_spotter import WSHyp
from nemo.utils import logging
import numpy as np
import json
from kaldialign import align
import texterrors


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
        if not isinstance(candidate.y_sequence, list):
            candidate.y_sequence = candidate.y_sequence.tolist()
        tokens = asr_model.tokenizer.ids_to_tokens(candidate.y_sequence)
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


def load_data(manifest: str) -> List[Dict]:
    """
    Load data from manifest file.

    Args:
        manifest: path to nemo manifest file.
    Returns:
        List of dicts with keys: audio_filepath, text, pred_text.
    """
    data = []
    with open(manifest, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data


def compute_fscore(recognition_results_manifest: str, key_words_list: List, return_scores: bool = False) -> Optional[tuple]:
    """
    Compute fscore for list of context biasing words/phrases.
    The idea is to get a word-level alignment for ground truth text and prediction results from manifest file.
    Then compute f-score for each word/phrase from key_words_list according to obtained word alignment.

    Args:
        recognition_results_manifest: path to nemo manifest file with recognition results in pred_text field.
        key_words_list: list of context biasing words/phrases.
        return_scores: if True, return precision, recall and fscore (not only print).
    Returns:
        If return_scores is True, return tuple of precision, recall and fscore.
    """
    
    assert key_words_list, "key_words_list is empty"

    # get data from manifest
    data = load_data(recognition_results_manifest)
    assert len(data) > 0, "manifest file is empty"
    assert data[0].get('text', None), "manifest file should contain text field"
    assert data[0].get('pred_text', None), "manifest file should contain pred_text field"

    # compute max number of words in one context biasing phrase
    max_ngram_order = max([len(item.split()) for item in key_words_list])
    key_words_stat = {} # a word here can be single word or phareses 
    for word in key_words_list:
        key_words_stat[word] = [0, 0, 0] # [true positive (tp), groud truth (gt), false positive (fp)]

    # auxiliary variable for epsilon token during alignment 
    eps = '***'

    for item in data:
        
        # texterrors
        ref = item['text'].split()
        hyp = item['pred_text'].split()
        texterrors_ali = texterrors.align_texts(ref, hyp, False)
        ali = []
        for i in range(len(texterrors_ali[0])):
            ali.append((texterrors_ali[0][i], texterrors_ali[1][i]))
        
        # # kaldialign
        # ref = item['text'].split()
        # hyp = item['pred_text'].split()
        # ali = align(ref, hyp, eps)

        for idx, pair in enumerate(ali):
            # check all the ngrams:
            # TODO: add epsilon skipping to ge more accurate results for phrases...
            for ngram_order in range(1, max_ngram_order+1):
                if (idx+ngram_order-1) < len(ali):
                    item_ref, item_hyp = [], []
                    for order in range(1, ngram_order+1):
                        item_ref.append(ali[idx+order-1][0])
                        item_hyp.append(ali[idx+order-1][1])
                    item_ref = " ".join(item_ref)
                    item_hyp = " ".join(item_hyp)
                    # update key_words_stat
                    if item_ref in key_words_stat:
                        key_words_stat[item_ref][1] += 1 # add to gt
                        if item_ref == item_hyp:
                            key_words_stat[item_ref][0] += 1 # add to tp
                    elif item_hyp in key_words_stat:
                        key_words_stat[item_hyp][2] += 1 # add to fp
                else:
                    break
    
    tp = sum([key_words_stat[x][0] for x in key_words_stat])
    gt = sum([key_words_stat[x][1] for x in key_words_stat])
    fp = sum([key_words_stat[x][2] for x in key_words_stat])

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (gt + 1e-8)
    fscore = 2*(precision*recall)/(precision+recall + 1e-8)

    logging.info("============================================================")
    logging.info("Per words statistic (word: correct/totall | false positive):\n")
    max_len = max([len(x) for x in key_words_stat if key_words_stat[x][1] > 0 or key_words_stat[x][2] > 0])
    for word in key_words_list:
        if key_words_stat[word][1] > 0 or key_words_stat[word][2] > 0:
            false_positive = ""
            if key_words_stat[word][2] > 0:
                false_positive = key_words_stat[word][2]
            logging.info(f"{word:>{max_len}}: {key_words_stat[word][0]:3}/{key_words_stat[word][1]:<3} |{false_positive:>3}")
    logging.info("============================================================")
    logging.info("============================================================")
    logging.info(f"Precision: {precision:.4f} ({tp}/{tp + fp}) fp:{fp}")
    logging.info(f"Recall:    {recall:.4f} ({tp}/{gt})")
    logging.info(f"Fscore:    {fscore:.4f}")
    logging.info("============================================================")

    if return_scores:
        return (precision, recall, fscore)