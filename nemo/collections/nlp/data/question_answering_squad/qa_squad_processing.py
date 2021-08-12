# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import collections
import json
from typing import Dict, List, Optional

from tqdm import tqdm
from transformers.models.bert.tokenization_bert import BasicTokenizer

from nemo.collections.nlp.data.data_utils import (
    DataProcessor,
    check_chinese_char,
    is_whitespace,
    normalize_answer,
    normalize_chinese_answer,
)
from nemo.utils import logging

"""
Utility functions for Question Answering NLP tasks
Some parts of this code were adapted from the HuggingFace library at
https://github.com/huggingface/transformers
"""

TRAINING_MODE = "train"
EVALUATION_MODE = "eval"
INFERENCE_MODE = "infer"


def _get_tokens(s):
    """get normalized tokens"""
    if not s:
        return []
    return normalize_answer(s).split()


def _get_tokens(s):
    """get normalized tokens for both Chinese and English"""
    if not s:
        return []

    # separate answers to en and ch pieces
    ch_seq = ""
    en_seq = ""
    pos = 0

    # Normalize and connect
    final_tokens = []

    while pos < len(s):
        if check_chinese_char(s[pos]):
            if en_seq != "":
                final_tokens.extend(normalize_answer(en_seq).split())
                en_seq = ""
            ch_seq += s[pos]
        else:
            if ch_seq != "":
                final_tokens.extend(normalize_chinese_answer(ch_seq))
                ch_seq = ""
            en_seq += s[pos]
        pos += 1

    if en_seq != "":
        final_tokens.extend(normalize_answer(en_seq).split())

    if ch_seq != "":
        final_tokens.extend(normalize_chinese_answer(ch_seq))

    return final_tokens


def get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def get_final_text(pred_text: str, orig_text: str, do_lower_case: bool, verbose_logging: bool = False):
    """Project the tokenized prediction back to the original text.
    When we created the data, we kept track of the alignment between original
    (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    now `orig_text` contains the span of our original text corresponding to
    the span that we predicted.

    However, `orig_text` may contain extra characters that we don't want in
    our prediction.

    For example, let's say:
      pred_text = steve smith
      orig_text = Steve Smith's

    We don't want to return `orig_text` because it contains the extra "'s".

    We don't want to return `pred_text` because it's already been normalized
    (the SQuAD eval script also does punctuation stripping/lower casing but
    our tokenizer does additional normalization like stripping accent
    characters).

    What we really want to return is "Steve Smith".

    Therefore, we have to apply a semi-complicated alignment heuristic
    between `pred_text` and `orig_text` to get a character-to-character
    alignment. This can fail in certain cases in which case we just return
    `orig_text`."""

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return ns_text, ns_to_s_map

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logging.warning("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logging.warning(
                "Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text,
            )
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logging.warning("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logging.warning("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text


def f1_score(prediction, ground_truth):
    """computes f1 score between prediction and ground truth"""
    prediction_tokens = _get_tokens(prediction)
    ground_truth_tokens = _get_tokens(ground_truth)
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
    """computes exact match between prediction and ground truth"""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    """Applies no answer threshhold"""
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    """returns dictionary with formatted evaluation scores"""
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
    """merges 2 evaluation dictionaries into the first one by adding prefix as key for name collision handling"""
    for k in new_eval:
        main_eval["%s_%s" % (prefix, k)] = new_eval[k]


def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    """
    Find best threshholds to maximize all evaluation metrics.
    """
    best_exact, exact_thresh = _find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = _find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)

    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh


def _find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    """
    Find best threshhold to maximize evaluation metric
    """
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


class SquadProcessor(DataProcessor):
    """
    Processor for the SQuAD data set.
    used by the version 1.1 and version 2.0 of SQuAD, respectively.

    Args:
        data_file: data file path
        mode: TRAINING_MODE/EVALUATION_MODE/INFERENCE_MODE for creating training/evaluation/inference dataset
    """

    def __init__(self, data_file: str, mode: str):
        self.data_file = data_file
        self.mode = mode

    def get_examples(self):
        if self.data_file is None:
            raise ValueError(f"{self.mode} data file is None.")

        with open(self.data_file, "r", encoding="utf-8") as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, set_type=self.mode)

    def _create_examples(self, input_data, set_type):
        examples = []
        for entry in tqdm(input_data):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []
                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        if set_type in [TRAINING_MODE, EVALUATION_MODE]:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        if set_type == EVALUATION_MODE:
                            answers = qa["answers"]

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )

                    examples.append(example)
        return examples


class SquadExample(object):
    """
    A single training/test example for the Squad dataset, as loaded from disk.
    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of
            the answer, 0 indexed
        title: The title of the example
        answers: None by default, this is used during evaluation.
            Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has
            no possible answer.
    """

    def __init__(
        self,
        qas_id: str,
        question_text: str,
        context_text: str,
        answer_text: str,
        start_position_character: int,
        title: str,
        answers: List[str] = [],
        is_impossible: bool = False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens
        # may be attributed to their original position.
        # ex: context_text = ["hi yo"]
        #     char_to_word_offset = [0, 0, 0, 1, 1]
        #     doc_tokens = ["hi", "yo"]
        for c in self.context_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start end end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            # start_position is index of word, end_position inclusive
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]


def convert_examples_to_features(
    examples: List[object],
    tokenizer: object,
    max_seq_length: int,
    doc_stride: int,
    max_query_length: int,
    has_groundtruth: bool,
):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.text_to_tokens(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        # context: index of token -> index of word
        tok_to_orig_index = []
        # context: index of word -> index of first token in token list
        orig_to_tok_index = []
        # context without white spaces after tokenization
        all_doc_tokens = []
        # doc tokens is word separated context
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.text_to_tokens(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        # idx of query token start and end in context
        tok_start_position = None
        tok_end_position = None
        if has_groundtruth and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if has_groundtruth and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
            )
        # The -3 accounts for tokenizer.cls_token, tokenizer.sep_token and tokenizer.sep_token
        # doc_spans contains all possible contexts options of given length
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            # maps context tokens idx in final input -> word idx in context
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append(tokenizer.cls_token)
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append(tokenizer.sep_token)
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append(tokenizer.sep_token)
            segment_ids.append(1)

            input_ids = tokenizer.tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens.
            # Only real tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(tokenizer.pad_id)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            # calculate start and end position in final array
            # of tokens in answer if no answer,
            # 0 for both pointing to tokenizer.cls_token
            start_position = None
            end_position = None
            if has_groundtruth and not example.is_impossible:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
            if has_groundtruth and example.is_impossible:
                # if our document chunk does not contain
                # an annotation we throw it out, since there is nothing
                # to predict.
                start_position = 0
                end_position = 0

            if example_index < 1:
                logging.info("*** Example ***")
                logging.info("unique_id: %s" % (unique_id))
                logging.info("example_index: %s" % (example_index))
                logging.info("doc_span_index: %s" % (doc_span_index))
                logging.info("tokens: %s" % " ".join(tokens))
                logging.info(
                    "token_to_orig_map: %s" % " ".join(["%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()])
                )
                logging.info(
                    "token_is_max_context: %s"
                    % " ".join(["%d:%s" % (x, y) for (x, y) in token_is_max_context.items()])
                )
                logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if has_groundtruth and example.is_impossible:
                    logging.info("impossible example")
                if has_groundtruth and not example.is_impossible:
                    answer_text = " ".join(tokens[start_position : (end_position + 1)])
                    logging.info("start_position: %d" % (start_position))
                    logging.info("end_position: %d" % (end_position))
                    logging.info("answer: %s" % (answer_text))
            if example_index % 100 == 0:
                logging.info(f"Finished processing: {example_index}")
            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible,
                )
            )
            unique_id += 1

    return features


def _improve_answer_span(
    doc_tokens: List[str], input_start: int, input_end: int, tokenizer: object, orig_answer_text: str
):
    """Returns tokenized answer spans that
    better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.text_to_tokens(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token.
    Because of the sliding window approach taken to scoring documents,
    a single token can appear in multiple documents.
    Example:
        Doc: the man went to the store and bought a gallon of milk
        Span A: the man went to the
        Span B: to the store and bought
        Span C: and bought a gallon of
        ...
    Now the word 'bought' will have two scores from spans B and C. We only
    want to consider the score with "maximum context", which we define as
    the *minimum* of its left and right context (the *sum* of left and
    right context will always be the same, of course).
    In the example the maximum context for 'bought' would be span C since
    it has 1 left context and 3 right context, while span B has 4 left context
    and 0 right context.
    Code adapted from the code by the Google AI and HuggingFace.
    """
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        unique_id: int,
        example_index: int,
        doc_span_index: int,
        tokens: List[str],
        token_to_orig_map: Dict[int, int],
        token_is_max_context: Dict[int, bool],
        input_ids: List[int],
        input_mask: List[int],
        segment_ids: List[int],
        start_position: Optional[int] = None,
        end_position: Optional[int] = None,
        is_impossible: Optional[int] = None,
    ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
