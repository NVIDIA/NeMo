"""
Copyright 2018 The Google AI Language Team Authors and
The HuggingFace Inc. team.
Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
__all__ = [
    'f1_score',
    'exact_match_score',
    'apply_no_ans_threshold',
    'make_eval_dict',
    'merge_eval',
    'find_all_best_thresh',
    'find_best_thresh',
]
import collections

from nemo.collections.nlp.data.datasets.datasets_utils import get_tokens, normalize_answer


def f1_score(prediction, ground_truth):
    prediction_tokens = get_tokens(prediction)
    ground_truth_tokens = get_tokens(ground_truth)
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if len(ground_truth_tokens) == 0 or len(prediction_tokens) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(ground_truth_tokens == prediction_tokens)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ("total", total),
            ]
        )


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval["%s_%s" % (prefix, k)] = new_eval[k]


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)

    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh


def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for _, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh
