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

import copy
import numpy as np
from nemo.utils import logging
from collections import deque
from typing import List, Optional, Dict
from context_graph_ctc import ContextState, ContextGraphCTC


class Token:
    """
    Class of alignment tracking according to the Token Passing Algoritm

    Args:
        state: state of Context-Biasing graph
        dist: accumulative score (or distance) of the token in log semiring
        start_frame: index of acoustic frame from which the token was created 
    """
    def __init__(
            self,
            state: ContextState,
            dist: float = 0.0,
            start_frame: Optional[int] = None
        ):
        self.state = state
        self.dist = dist    
        self.start_frame = start_frame 
        self.alive = True


class WSHyp:
    """
    Hypotheis for Word Spotter

    Args:
        word: spotter word
        score: accumulative score (distance) of best token
        start_frame: index of acoustic frame from which the best token was created
        end_frame: index of acoustic frame from which the final state of ContextGraph was reached
        tokenization: tokenization of spotted word
    """
    def __init__(
            self,
            word: str,
            score: float,
            start_frame: int,
            end_frame: int,
            tokenization: List[int]
        ):
        self.word = word
        self.score = score
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.tokenization = tokenization
        

def beam_pruning(next_tokens: List[Token], beam_threshold: float) -> List[Token]:
    """ Prun all tokens whose dist is worse than best_token.dist - beam_threshold """
    if not next_tokens:
        return []   
    best_token = next_tokens[np.argmax([token.dist for token in next_tokens])]
    next_tokens = [token for token in next_tokens if token.dist > best_token.dist - beam_threshold]
    return next_tokens


def state_pruning(next_tokens: List[Token]) -> List[Token]:
    """ If there are several tokens on one state, then leave only the best of them by distance"""
    if not next_tokens:
        return []
    # hyps pruning
    for token in next_tokens:
        if not token.state.best_token:
            token.state.best_token = token
        else:
            if token.dist <= token.state.best_token.dist:
                token.alive = False
            else:
                token.state.best_token.alive = False
                token.state.best_token = token
    next_tokens_pruned = [token for token in next_tokens if token.alive]
    # clean best_tokens in context_graph
    for token in next_tokens:
        token.state.best_token = None
    return next_tokens_pruned


def find_best_hyp(spotted_words: List[WSHyp], intersection_thershold: int = 10) -> List[WSHyp]:
    """
    Some spotted hypotheses may have overlap.
    If hypotheses intersection is greater than intersection_thershold,
    then the function leaves only the best hypothesis according to the score.
    """

    hyp_intervals_dict = {}
    for hyp in spotted_words:
        hyp_interval = set(range(hyp.start_frame, hyp.end_frame+1))
        h_interval_name = f"{hyp.start_frame}_{hyp.end_frame}"
        insert_new_hyp = True

        for h_interval_key in hyp_intervals_dict:
            cl, cr = int(h_interval_key.split("_")[0]), int(h_interval_key.split("_")[1])
            cluster_interval = set(range(cl, cr+1))
            intersection_part = 100/len(cluster_interval) * len(hyp_interval & cluster_interval)
            # in case of intersection:
            if intersection_part >= intersection_thershold:
                if hyp.score > hyp_intervals_dict[h_interval_key].score:
                    hyp_intervals_dict.pop(h_interval_key)
                    insert_new_hyp = True
                    break
                else:
                    insert_new_hyp = False         
        if insert_new_hyp:
            hyp_intervals_dict[h_interval_name] = hyp
    
    best_hyp_list = [hyp_intervals_dict[h_interval_key] for h_interval_key in hyp_intervals_dict]        
    
    return best_hyp_list


def get_ctc_word_alignment(logprob: np.ndarray, asr_model, token_weight: float = 1.0) -> List[tuple]:
    """ 
    Get word level alignment (with start and end frames) based on argmax ctc predictions.
    The word score is a sum of non-blank token logprobs with additional token_weight.
    token_weight is used to prevent false accepts during context biasing
    """
    
    alignment_ctc = np.argmax(logprob, axis=1)

    # get token alignment
    token_alignment = []
    prev_idx = None
    for i, idx in enumerate(alignment_ctc):
        token_logprob = 0
        if idx != asr_model.decoder.blank_idx:
            token = asr_model.tokenizer.ids_to_tokens([int(idx)])[0]
            if idx == prev_idx:
                prev_repited_token = token_alignment.pop()
                token_logprob += prev_repited_token[2]
            token_logprob += logprob[i, idx].item()
            token_alignment.append((token, i, token_logprob))
        prev_idx = idx
    
    # get word alignment
    slash = "▁"
    word_alignment = []
    word = ""
    l, r, score = None, None, None
    for item in token_alignment:
        if not word:
            if word.startswith(slash):
                word = item[0][1:]
            else:
                word = item[0][:]
            l = item[1]
            r = item[1]
            score = item[2]+token_weight
        else:
            if item[0].startswith(slash):
                word_alignment.append((word, l, r, score))
                word = item[0][1:]
                l = item[1]
                r = item[1]
                score = item[2]+token_weight
            else:
                word += item[0]
                r = item[1]
                score += item[2]+token_weight
    if word:
        word_alignment.append((word, l, r, score))
    
    if len(word_alignment) == 1 and not word_alignment[0][0]:
        word_alignment = []
    
    return word_alignment


def filter_wb_hyps(best_hyp_list: List[WSHyp], word_alignment: List[tuple]) -> List[WSHyp]:
    """
    Compare scores of spotted words with overlapping words from ctc alignment.
    If score of spotted word is less than overalapping words from ctc alignment,
    the spotted word will removed as false positive.
    A spotted word may overlap with several words from ctc alignment ("gpu" -> "g p u").
    Here we use overall_spot_score variable to accomulate scores of several words.
    """
    
    if not word_alignment:
        return best_hyp_list

    best_hyp_list_filtered = []
    current_word_in_ali = 0
    for hyp in best_hyp_list:
        overall_spot_score = 0
        word_spotted = False
        hyp_interval = set(range(hyp.start_frame, hyp.end_frame+1))
        for i in range(current_word_in_ali, len(word_alignment)):
            item = word_alignment[i]
            item_interval = set(range(item[1], item[2]+1))
            intersection_part = 100/len(item_interval) * len(hyp_interval & item_interval)
            if intersection_part:
                if not word_spotted:
                    overall_spot_score = item[3]
                else:
                    overall_spot_score += intersection_part/100 * item[3]
                word_spotted = True
            elif word_spotted:
                if hyp.score >= overall_spot_score:
                    best_hyp_list_filtered.append(hyp)
                    current_word_in_ali = i
                    word_spotted = False
                    break
        if word_spotted and hyp.score >= overall_spot_score:
            best_hyp_list_filtered.append(hyp)

    return best_hyp_list_filtered


def run_word_spotter(
        logprobs: np.ndarray,
        context_graph: ContextGraphCTC,
        asr_model,
        beam_threshold: float = 5.0,
        cb_weight: float = 3.0,
        ctc_ali_token_weight: float = 0.5,
        keyword_threshold: float = -5.0,
        blank_threshold: float = 0.8,
        print_results: bool = False
    ):
    """
    CTC-based Word Spotter for recognition of words from context biasing graph (paper link) 
    
    Args:
       logprobs: CTC logprobs
       context_graph: Context-Biasing graph
       asr_model: ASR model (ctc or hybrid-transducer-ctc)
       beam_threshold: threshold for beam pruning
       cb_weight: context biasing weight
       ctc_ali_token_weight: additional token weight for word-level ctc alignment
       keyword_threshold: auxiliary weight for pruning final hypotheses
       blank_threshold: blank threshold (probability) for preliminary hypotheses pruning
       print_results: print spotted words if True
    """


    start_state = context_graph.root
    active_tokens = []
    next_tokens = []
    spotted_words = []

    # move probability to log space
    blank_threshold = np.log(blank_threshold)
    blank_idx = asr_model.decoder.blank_idx

    for frame in range(logprobs.shape[0]):
        # add an empty token (located in the graph root) at each new frame to start new word spotting
        active_tokens.append(Token(start_state, start_frame=frame))
        logprob_frame = logprobs[frame]
        best_dist = None
        for token in active_tokens:
            # skip token by the blank threshold if empty token
            if token.state is context_graph.root and logprobs[frame][blank_idx] > blank_threshold:
                continue
            for transition_state in token.state.next:
                # running beam pruning -- allows to skip the number of Token class creations
                if transition_state != blank_idx:
                    # add cb_weight only for non-blank tokens
                    current_dist = token.dist + logprob_frame[int(transition_state)].item() + cb_weight
                else:
                    current_dist = token.dist + logprob_frame[int(transition_state)].item()
                if not best_dist:
                    best_dist = current_dist
                else:
                    if current_dist < best_dist - beam_threshold:
                        continue
                    elif current_dist > best_dist:
                        best_dist = current_dist

                new_token = Token(token.state.next[transition_state], current_dist, token.start_frame)
                # if transition_state != asr_model.decoder.blank_idx:
                #     new_token.non_blank_score += logprob_frame[int(transition_state)].item() + context_score

                # if not new_token.start_frame:
                #     new_token.start_frame = frame

                # if end of word:
                if new_token.state.is_end and new_token.dist > keyword_threshold:
                    # word = asr_model.tokenizer.ids_to_text(new_token.state.word)
                    word = new_token.state.word
                    spotted_words.append(WSHyp(word, new_token.dist, new_token.start_frame, frame, asr_model.tokenizer.text_to_ids(new_token.state.word)))
                    # spotted_words.append(WBHyp(word, new_token.non_blank_score, new_token.start_frame, frame, new_token.state.word))
                    if len(new_token.state.next) == 1:
                        if current_dist is best_dist:
                            best_dist = None
                        continue
                next_tokens.append(new_token)
                # else:
                #     next_tokens.append(new_token)
        # state and beam prunings:
        next_tokens = beam_pruning(next_tokens, beam_threshold)
        next_tokens = state_pruning(next_tokens)
        # print(f"frame step is: {frame}")
        # print(f"number of active_tokens is: {len(next_tokens)}")

        active_tokens = next_tokens
        next_tokens = []

    # find best hyp for spotted keywords:
    best_hyp_list = find_best_hyp(spotted_words)
    if print_results:
        print(f"---spotted words:")
        for hyp in best_hyp_list:
            print(f"{hyp.word}: [{hyp.start_frame};{hyp.end_frame}], score:{hyp.score:-.2f}")
    
    # filter wb hyps according to greedy ctc predictions
    ctc_word_alignment = get_ctc_word_alignment(logprobs, asr_model, token_weight=ctc_ali_token_weight)
    best_hyp_list_new = filter_wb_hyps(best_hyp_list, ctc_word_alignment)
    if print_results:
        print("---final result is:")
        for hyp in best_hyp_list_new:
            print(f"{hyp.word}: [{hyp.start_frame};{hyp.end_frame}], score:{hyp.score:-.2f}")


    return best_hyp_list_new