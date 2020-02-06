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
import collections
import json
import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from nemo import logging
from nemo.collections.nlp.data.datasets.glue_benchmark_dataset import DataProcessor
from nemo.collections.nlp.metrics.squad_metrics import (
    _get_best_indexes,
    apply_no_ans_threshold,
    exact_match_score,
    f1_score,
    find_all_best_thresh,
    get_final_text,
    make_eval_dict,
    merge_eval,
    normalize_answer,
)
from nemo.collections.nlp.utils.common_nlp_utils import _is_whitespace
from nemo.collections.nlp.utils.loss_utils import _compute_softmax

__all__ = ['SquadDataset']

"""
Utility functions for Question Answering NLP tasks
Some parts of this code were adapted from the HuggingFace library at
https://github.com/huggingface/transformers
"""


class SquadDataset(Dataset):
    """
    Creates SQuAD dataset for Question Answering.

    Args:
        data_file (str): train.*.json or dev.*.json.
        tokenizer (obj): Tokenizer object, e.g. NemoBertTokenizer.
        version_2_with_negative (bool): True if training should allow
            unanswerable questions.
        doc_stride (int): When splitting up a long document into chunks,
            how much stride to take between chunks.
        max_query_length (iny): All training files which have a duration less
            than min_duration are dropped. Can't be used if the `utt2dur` file
            does not exist. Defaults to None.
        max_seq_length (int): All training files which have a duration more
            than max_duration are dropped. Can't be used if the `utt2dur` file
            does not exist. Defaults to None.
        mode (str): Use "train" or "dev" to define between
            training and evaluation.
    """

    def __init__(
        self, data_file, tokenizer, doc_stride, max_query_length, max_seq_length, version_2_with_negative, mode
    ):
        self.tokenizer = tokenizer
        self.version_2_with_negative = version_2_with_negative
        self.processor = SquadProcessor(data_file=data_file, mode=mode)
        self.mode = mode
        if mode != "dev" and mode != "train":
            raise ValueError(f"mode should be either 'train' or 'dev' but got {mode}")
        self.examples = self.processor.get_examples()

        if mode == "train":
            cached_train_features_file = (
                data_file
                + '_cache'
                + '_{0}_{1}_{2}_{3}'.format(mode, str(max_seq_length), str(doc_stride), str(max_query_length))
            )

            if os.path.exists(cached_train_features_file):
                with open(cached_train_features_file, "rb") as reader:
                    self.features = pickle.load(reader)
            else:
                self.features = convert_examples_to_features(
                    examples=self.examples,
                    tokenizer=tokenizer,
                    max_seq_length=max_seq_length,
                    doc_stride=doc_stride,
                    max_query_length=max_query_length,
                    has_groundtruth=True,
                )
                master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
                if master_device:
                    logging.info("  Saving train features into cached file %s", cached_train_features_file)
                    with open(cached_train_features_file, "wb") as writer:
                        pickle.dump(self.features, writer)
        elif mode == "dev":
            self.features = convert_examples_to_features(
                examples=self.examples,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                doc_stride=doc_stride,
                max_query_length=max_query_length,
                has_groundtruth=True,
            )
        else:
            raise Exception

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        return (
            np.array(feature.input_ids),
            np.array(feature.segment_ids),
            np.array(feature.input_mask),
            np.array(feature.start_position),
            np.array(feature.end_position),
            np.array(feature.unique_id),
        )

    def get_predictions(
        self,
        unique_ids,
        start_logits,
        end_logits,
        n_best_size,
        max_answer_length,
        do_lower_case,
        version_2_with_negative,
        null_score_diff_threshold,
    ):
        example_index_to_features = collections.defaultdict(list)

        unique_id_to_pos = {}
        for index, unique_id in enumerate(unique_ids):
            unique_id_to_pos[unique_id] = index

        for feature in self.features:
            example_index_to_features[feature.example_index].append(feature)

        _PrelimPrediction = collections.namedtuple(
            "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
        )

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        scores_diff_json = collections.OrderedDict()

        for (example_index, example) in enumerate(self.examples):

            features = example_index_to_features[example_index]

            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            # large and positive
            score_null = 1000000
            # the paragraph slice with min null score
            min_null_feature_index = 0
            # start logit at the slice with min null score
            null_start_logit = 0
            # end logit at the slice with min null score
            null_end_logit = 0
            for (feature_index, feature) in enumerate(features):
                pos = unique_id_to_pos[feature.unique_id]
                start_indexes = _get_best_indexes(start_logits[pos], n_best_size)
                end_indexes = _get_best_indexes(end_logits[pos], n_best_size)
                # if we could have irrelevant answers,
                # get the min score of irrelevant
                if version_2_with_negative:
                    feature_null_score = start_logits[pos][0] + end_logits[pos][0]
                    if feature_null_score < score_null:
                        score_null = feature_null_score
                        min_null_feature_index = feature_index
                        null_start_logit = start_logits[pos][0]
                        null_end_logit = end_logits[pos][0]
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # We could hypothetically create invalid predictions,
                        # e.g., predict that the start of the span is in the
                        # question. We throw out all invalid predictions.
                        if start_index >= len(feature.tokens):
                            continue
                        if end_index >= len(feature.tokens):
                            continue
                        if start_index not in feature.token_to_orig_map:
                            continue
                        if end_index not in feature.token_to_orig_map:
                            continue
                        if not feature.token_is_max_context.get(start_index, False):
                            continue
                        if end_index < start_index:
                            continue
                        length = end_index - start_index + 1
                        if length > max_answer_length:
                            continue
                        prelim_predictions.append(
                            _PrelimPrediction(
                                feature_index=feature_index,
                                start_index=start_index,
                                end_index=end_index,
                                start_logit=start_logits[pos][start_index],
                                end_logit=end_logits[pos][end_index],
                            )
                        )

            if version_2_with_negative:
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=min_null_feature_index,
                        start_index=0,
                        end_index=0,
                        start_logit=null_start_logit,
                        end_logit=null_end_logit,
                    )
                )
            prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

            _NbestPrediction = collections.namedtuple("NbestPrediction", ["text", "start_logit", "end_logit"])

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break
                feature = features[pred.feature_index]
                if pred.start_index > 0:  # this is a non-null prediction
                    tok_tokens = feature.tokens[pred.start_index : (pred.end_index + 1)]
                    orig_doc_start = feature.token_to_orig_map[pred.start_index]
                    orig_doc_end = feature.token_to_orig_map[pred.end_index]
                    orig_tokens = example.doc_tokens[orig_doc_start : (orig_doc_end + 1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    final_text = get_final_text(tok_text, orig_text, do_lower_case)
                    if final_text in seen_predictions:
                        continue

                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True

                nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))
            # if we didn't include the empty option in the n-best, include it
            if version_2_with_negative:
                if "" not in seen_predictions:
                    nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit))

                # In very rare edge cases we could only
                # have single null pred. We just create a nonce prediction
                # in this case to avoid failure.
                if len(nbest) == 1:
                    nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

            assert len(nbest) >= 1

            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)
                if not best_non_null_entry:
                    if entry.text:
                        best_non_null_entry = entry

            probs = _compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                nbest_json.append(output)

            assert len(nbest_json) >= 1

            if not version_2_with_negative:
                all_predictions[example.qas_id] = nbest_json[0]["text"]
            else:
                # predict "" iff the null score -
                # the score of best non-null > threshold
                score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
                scores_diff_json[example.qas_id] = score_diff
                if score_diff > null_score_diff_threshold:
                    all_predictions[example.qas_id] = ""
                else:
                    all_predictions[example.qas_id] = best_non_null_entry.text
                all_nbest_json[example.qas_id] = nbest_json

        return all_predictions, all_nbest_json, scores_diff_json

    def evaluate_predictions(self, all_predictions, no_answer_probs=None, no_answer_probability_threshold=1.0):
        qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in self.examples}
        has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
        no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]
        if no_answer_probs is None:
            no_answer_probs = {k: 0.0 for k in all_predictions}

        exact, f1 = self.get_raw_scores(all_predictions)

        exact_threshold = apply_no_ans_threshold(
            exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
        )
        f1_threshold = apply_no_ans_threshold(
            f1, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
        )

        evaluation = make_eval_dict(exact_threshold, f1_threshold)

        if has_answer_qids:
            has_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=has_answer_qids)
            merge_eval(evaluation, has_ans_eval, "HasAns")

        if no_answer_qids:
            no_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=no_answer_qids)
            merge_eval(evaluation, no_ans_eval, "NoAns")

        if no_answer_probs:
            find_all_best_thresh(evaluation, all_predictions, exact, f1, no_answer_probs, qas_id_to_has_answer)

        return evaluation["best_exact"], evaluation["best_f1"]

    def get_raw_scores(self, preds):
        """
        Computes the exact and f1 scores from the examples
        and the model predictions
        """
        exact_scores = {}
        f1_scores = {}

        for example in self.examples:
            qas_id = example.qas_id
            gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]

            if not gold_answers:
                # For unanswerable questions,
                # only correct answer is empty string
                gold_answers = [""]

            if qas_id not in preds:
                logging.warning("Missing prediction for %s" % qas_id)
                continue

            prediction = preds[qas_id]
            exact_scores[qas_id] = max(exact_match_score(a, prediction) for a in gold_answers)
            f1_scores[qas_id] = max(f1_score(a, prediction) for a in gold_answers)

        return exact_scores, f1_scores

    def evaluate(
        self,
        unique_ids,
        start_logits,
        end_logits,
        n_best_size,
        max_answer_length,
        do_lower_case,
        version_2_with_negative,
        null_score_diff_threshold,
    ):

        (all_predictions, all_nbest_json, scores_diff_json) = self.get_predictions(
            unique_ids,
            start_logits,
            end_logits,
            n_best_size,
            max_answer_length,
            do_lower_case,
            version_2_with_negative,
            null_score_diff_threshold,
        )

        exact_match, f1 = self.evaluate_predictions(all_predictions)

        return exact_match, f1, all_predictions


def convert_examples_to_features(examples, tokenizer, max_seq_length, doc_stride, max_query_length, has_groundtruth):
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

        # The -3 accounts for [CLS], [SEP] and [SEP]
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
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens.
            # Only real tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            # calculate start and end position in final array
            # of tokens in answer if no answer,
            # 0 for both pointing to [CLS]
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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        unique_id,
        example_index,
        doc_span_index,
        tokens,
        token_to_orig_map,
        token_is_max_context,
        input_ids,
        input_mask,
        segment_ids,
        start_position=None,
        end_position=None,
        is_impossible=None,
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


class SquadProcessor(DataProcessor):
    """
    Processor for the SQuAD data set.
    used by the version 1.1 and version 2.0 of SQuAD, respectively.
    """

    def __init__(self, data_file, mode):
        self.data_file = data_file
        self.mode = mode

    def get_examples(self):
        if self.data_file is None:
            raise ValueError("SquadProcessor should be instantiated")

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
                        if set_type == "train" or set_type == "dev":
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        if set_type == "dev":
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
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        answers=[],
        is_impossible=False,
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
            if _is_whitespace(c):
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


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
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
    """Check if this is the 'max context' doc span for the token."""
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


def check_is_max_context(doc_spans, cur_span_index, position):
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
