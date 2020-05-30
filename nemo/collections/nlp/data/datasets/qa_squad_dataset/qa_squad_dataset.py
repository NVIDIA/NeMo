# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
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
# =============================================================================

import collections
import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from nemo import logging
from nemo.collections.nlp.data.datasets.datasets_utils.data_preprocessing import is_whitespace
from nemo.collections.nlp.data.datasets.datasets_utils.datasets_processing import DataProcessor
from nemo.collections.nlp.data.datasets.qa_squad_dataset.qa_squad_processing import convert_examples_to_features
from nemo.collections.nlp.metrics.squad_metrics import (
    _get_best_indexes,
    apply_no_ans_threshold,
    exact_match_score,
    f1_score,
    find_all_best_thresh,
    get_final_text,
    make_eval_dict,
    merge_eval,
)
from nemo.collections.nlp.utils.data_utils import normalize_answer
from nemo.collections.nlp.utils.functional_utils import _compute_softmax

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
        data_file (str): train.*.json eval.*.json or test.*.json.
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
        mode (str): Use "train", "eval" or "test" to define between
            training and evaluation.
    """

    def __init__(
        self,
        data_file,
        tokenizer,
        doc_stride,
        max_query_length,
        max_seq_length,
        version_2_with_negative,
        mode,
        use_cache,
    ):
        self.tokenizer = tokenizer
        self.version_2_with_negative = version_2_with_negative
        self.processor = SquadProcessor(data_file=data_file, mode=mode)
        self.mode = mode
        if mode not in ["eval", "train", "test"]:
            raise ValueError(f"mode should be either 'train', 'eval', or 'test' but got {mode}")
        self.examples = self.processor.get_examples()

        tokenizer_type = type(tokenizer.tokenizer).__name__
        vocab_size = getattr(tokenizer, "vocab_size", 0)
        cached_features_file = (
            data_file
            + '_cache'
            + '_{}_{}_{}_{}_{}_{}'.format(
                mode, tokenizer_type, str(vocab_size), str(max_seq_length), str(doc_stride), str(max_query_length)
            )
        )

        if use_cache and os.path.exists(cached_features_file):
            logging.info(f"loading from {cached_features_file}")
            with open(cached_features_file, "rb") as reader:
                self.features = pickle.load(reader)
        else:
            self.features = convert_examples_to_features(
                examples=self.examples,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                doc_stride=doc_stride,
                max_query_length=max_query_length,
                has_groundtruth=mode != "test",
            )

            if use_cache:
                master_device = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
                if master_device:
                    logging.info("  Saving train features into cached file %s", cached_features_file)
                    with open(cached_features_file, "wb") as writer:
                        pickle.dump(self.features, writer)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        if self.mode == "test":
            return (
                np.array(feature.input_ids),
                np.array(feature.segment_ids),
                np.array(feature.input_mask),
                np.array(feature.unique_id),
            )
        else:
            return (
                np.array(feature.input_ids),
                np.array(feature.segment_ids),
                np.array(feature.input_mask),
                np.array(feature.unique_id),
                np.array(feature.start_position),
                np.array(feature.end_position),
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
                output["start_logit"] = (
                    entry.start_logit if isinstance(entry.start_logit, float) else list(entry.start_logit)
                )
                output["end_logit"] = entry.end_logit if isinstance(entry.end_logit, float) else list(entry.end_logit)
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

        return exact_match, f1, all_predictions, all_nbest_json


class SquadProcessor(DataProcessor):
    """
    Processor for the SQuAD data set.
    used by the version 1.1 and version 2.0 of SQuAD, respectively.
    """

    def __init__(self, data_file, mode):
        self.data_file = data_file
        self.mode = mode

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
                        if set_type == "train" or set_type == "eval":
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        if set_type == "eval":
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
