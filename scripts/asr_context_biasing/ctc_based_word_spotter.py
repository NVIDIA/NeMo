import copy
import numpy as np
from nemo.utils import logging
from collections import deque


class Token:
    def __init__(self, state, dist=0.0, start_frame=None, non_blank_score=0.0):
        self.state = state
        self.dist = dist     
        self.alive = True
        self.start_frame = start_frame
        self.non_blank_score = non_blank_score

class WBHyp:
    def __init__(self, word, score, start_frame, end_frame, tokenization):
        self.word = word
        self.score = score
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.tokenization = tokenization
        

def beam_pruning(next_tokens, threshold):
    if not next_tokens:
        return []   
    # alive_tokens = [token for token in next_tokens if token.alive]
    best_token = next_tokens[np.argmax([token.dist for token in next_tokens])]
    next_tokens = [token for token in next_tokens if token.dist > best_token.dist - threshold]
    return next_tokens


def state_pruning(next_tokens):
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


def find_best_hyp(spotted_words):

    clusters_dict = {}
    for hyp in spotted_words:
        # hl, hr = hyp.start_frame, hyp.end_frame
        hyp_interval = set(range(hyp.start_frame, hyp.end_frame+1))
        h_cluster_name = f"{hyp.start_frame}_{hyp.end_frame}"
        insert_cluster = True

        for cluster in clusters_dict:
            cl, cr = int(cluster.split("_")[0]), int(cluster.split("_")[1])
            cluster_interval = set(range(cl, cr+1))
            intersection_part = 100/len(cluster_interval) * len(hyp_interval & cluster_interval)
            # in case of intersection:
            # TODO -- check if it the same word?
            if intersection_part >= 20:
                if hyp.score > clusters_dict[cluster].score:
                    clusters_dict.pop(cluster)
                    insert_cluster = True
                    break
                else:
                    insert_cluster = False         
        if insert_cluster:
            clusters_dict[h_cluster_name] = hyp
    
    best_hyp_list = [clusters_dict[cluster] for cluster in clusters_dict]        
    
    return best_hyp_list


def get_ctc_word_alignment(logprob, model, token_weight=1.0):
    
    alignment_ctc = np.argmax(logprob, axis=1)

    # get token alignment
    token_alignment = []
    prev_idx = None
    for i, idx in enumerate(alignment_ctc):
        token_logprob = 0
        if idx != model.decoder.blank_idx:
            token = model.tokenizer.ids_to_tokens([int(idx)])[0]
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
    token_boost = token_weight
    for item in token_alignment:
        if not word:
            if word.startswith(slash):
                word = item[0][1:]
            else:
                word = item[0][:]
            l = item[1]
            r = item[1]
            score = item[2]+token_boost
        else:
            if item[0].startswith(slash):
                word_alignment.append((word, l, r, score))
                word = item[0][1:]
                l = item[1]
                r = item[1]
                score = item[2]+token_boost
            else:
                word += item[0]
                r = item[1]
                score += item[2]+token_boost
    if word:
        word_alignment.append((word, l, r, score))
    
    if len(word_alignment) == 1 and not word_alignment[0][0]:
        word_alignment = []
    
    return word_alignment



def filter_wb_hyps(best_hyp_list, word_alignment):
    
    # logging.warning("---------------------")
    # logging.warning(f"word_alignment is: {word_alignment}")
    if not word_alignment:
        return best_hyp_list

    best_hyp_list_new = []
    current_word_in_ali = 0
    for hyp in best_hyp_list:
        # print("--------- wb candidat words: --------")
        # print(f"{hyp.word}: [{hyp.start_frame};{hyp.end_frame}], score:{hyp.score:-.2f}")
        overall_spot_score = 0
        overall_spot_word = ""
        word_spotted = False
        hyp_interval = set(range(hyp.start_frame, hyp.end_frame+1))
        # print("--------- spot information: --------")
        for i in range(current_word_in_ali, len(word_alignment)):
            item = word_alignment[i]
            item_interval = set(range(item[1], item[2]+1))
            intersection_part = 100/len(item_interval) * len(hyp_interval & item_interval)
            if intersection_part:
                if not word_spotted:
                    overall_spot_score = item[3]
                    # current_word_in_ali = i
                else:
                    overall_spot_score += intersection_part/100 * item[3]
                    # overall_spot_score += 0
                overall_spot_word += f"{item[0]} "
                word_spotted = True
                # print(item)
            elif word_spotted:
                if hyp.score >= overall_spot_score:
                    best_hyp_list_new.append(hyp)
                    current_word_in_ali = i
                    word_spotted = False
                    break
        if word_spotted and hyp.score >= overall_spot_score:
            best_hyp_list_new.append(hyp)


        # print(f"overal spot score: {overall_spot_score:.2f}")
        # print(f"overal spot word : {overall_spot_word}")

    return best_hyp_list_new


# def filter_wb_hyps(best_hyp_list, word_alignment):
    
#     # logging.warning("---------------------")
#     # logging.warning(f"word_alignment is: {word_alignment}")
#     if not word_alignment:
#         return best_hyp_list

#     best_hyp_list_new = []
#     current_frame = 0
#     for hyp in best_hyp_list:
#         lh, rh = hyp.start_frame, hyp.end_frame
#         for i in range(current_frame, len(word_alignment)):
#             item = word_alignment[i]
#             li, ri = item[1], item[2]
#             if li <= lh <= ri or li <= rh <= ri or lh <= li <= rh or lh <= ri <= rh:
#                 if hyp.score >= item[3]:
#                 # if hyp.score >= item[3] and not item[0].startswith(hyp.word):
#                     best_hyp_list_new.append(hyp)
#                 current_frame = i
#                 break
    
#     return best_hyp_list_new


# def filter_wb_hyps(best_hyp_list, word_alignment):
    
#     best_hyp_list_new = []
#     current_spot = 0
#     for hyp in best_hyp_list:
#         lh, rh = hyp.start_frame, hyp.end_frame
#         overall_spot_score = 0
#         spotted = False
#         for i in range(current_spot, len(word_alignment)):
#             item = word_alignment[i]
#             li, ri = item[1], item[2]
#             if li <= lh <= ri or li <= rh <= ri or lh <= li <= rh or lh <= ri <= rh:
#                 overall_spot_score += item[3]
#                 spotted = True
#             elif spotted and hyp.score >= overall_spot_score:
#                 best_hyp_list_new.append(hyp)
#                 current_spot = i-1
#                 break
    
#     return best_hyp_list_new


def recognize_wb(
        logprobs,
        context_graph,
        asr_model,
        beam_threshold=None,
        context_score=0.0,
        keyword_thr=-3,
        ctc_ali_token_weight=2.0,
        print_results=False
    ):

    start_state = context_graph.root
    active_tokens = []
    next_tokens = []
    spotted_words = []

    blank_thr = np.log(0.80)

    for frame in range(logprobs.shape[0]):
        active_tokens.append(Token(start_state, start_frame=frame))
        logprob_frame = logprobs[frame]
        best_dist = None
        for token in active_tokens:
            ## skip blank for first token if root:
            if token.state is context_graph.root and logprobs[frame][asr_model.decoder.blank_idx] > blank_thr:
                continue
            ## end skip blank
            for transition_state in token.state.next:

                ### running beam (start):
                if transition_state != asr_model.decoder.blank_idx:
                    # final_context_score = context_score / (token.state.next[transition_state].token_index+1)
                    final_context_score = context_score
                    current_dist = token.dist + \
                                   logprob_frame[int(transition_state)].item() + \
                                   final_context_score
                else:
                    current_dist = token.dist + logprob_frame[int(transition_state)].item()
                if not best_dist:
                    best_dist = current_dist
                else:
                    if current_dist < best_dist - beam_threshold:
                        continue
                    elif current_dist > best_dist:
                        best_dist = current_dist
                ### running beam (end)

                new_token = Token(token.state.next[transition_state], current_dist, token.start_frame, token.non_blank_score)
                # if transition_state != asr_model.decoder.blank_idx:
                #     new_token.non_blank_score += logprob_frame[int(transition_state)].item() + context_score

                # if not new_token.start_frame:
                #     new_token.start_frame = frame

                # if end of word:
                if new_token.state.is_end and new_token.dist > keyword_thr:
                    # word = asr_model.tokenizer.ids_to_text(new_token.state.word)
                    word = new_token.state.word
                    spotted_words.append(WBHyp(word, new_token.dist, new_token.start_frame, frame, asr_model.tokenizer.text_to_ids(new_token.state.word)))
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