# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
"""
Utility functions for Question/Answering NLP tasks
Some transformer of this code were adapted from the HuggingFace library at
https://github.com/huggingface/pytorch-pretrained-BERT
"""

#TODO: REFACTOR to minimize code reusing


import pickle
import collections
import json
import numpy as np
from torch.utils.data import Dataset
from ...externals.tokenization import whitespace_tokenize
from ...externals.tokenization import BasicTokenizer
from ...externals.run_squad import InputFeatures, \
    _compute_softmax, _get_best_indexes, _check_is_max_context
import string
import re
import torch


class BertQuestionAnsweringDataset(Dataset):
    def __init__(self,
                 input_file,
                 data_format,
                 features_file_prefix,
                 max_seq_length,
                 is_training,
                 tokenizer,
                 local_rank,
                 max_query_length=64,
                 doc_stride=128):

        # Read the context/question/answers from file with format data_format
        print("Reading examples")
        self.qa_examples = read_qa_examples(input_file, data_format,
                                            is_training)

        cached_features_file = input_file + '_{0}_{1}_{2}_{3}'.format(
            features_file_prefix, str(max_seq_length), str(doc_stride),
            str(max_query_length))

        print("Trying to open cached_features file:", cached_features_file)
        try:
            with open(cached_features_file, "rb") as reader:
                self.qa_features = pickle.load(reader)
        except:
            print("Converting examples to features")
            self.qa_features = convert_examples_to_features(
                self.qa_examples, tokenizer, max_seq_length, doc_stride,
                max_query_length, is_training)

            print("  Saving features into cached file %s",
                  cached_features_file)
            if (local_rank == None or torch.distributed.get_rank() == 0):
                print(f"Local rank {local_rank} writing cached file")
                with open(cached_features_file, "wb") as writer:
                    pickle.dump(self.qa_features, writer)
        #
        print("Number of features=", len(self.qa_features))

        #  #TODO: Deal with distributed training
        #  logger.info("  Saving train features into cached file %s",
        #  cached_train_features_file)
        #  with open(cached_train_features_file, "wb") as writer:
        #      pickle.dump(self.qa_features, writer)

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.vocab_size = self.tokenizer.vocab_size
        self.max_query_length = max_query_length
        self.doc_stride = doc_stride
        self.is_training = is_training

    def __len__(self):
        return len(self.qa_features)

    def __getitem__(self, idx):

        feature = self.qa_features[idx]

        return np.array(feature.input_ids), np.array(feature.segment_ids), \
               np.array(feature.input_mask, dtype=np.float32)[..., None], \
               feature.start_position, feature.end_position, feature.unique_id

    def get_predictions(self, unique_ids, start_logits, end_logits,
                        n_best_size, max_answer_length, do_lower_case,
                        version_2_with_negative, null_score_diff_threshold):

        example_index_to_features = collections.defaultdict(list)

        unique_id_to_pos = {}
        for index, unique_id in enumerate(unique_ids):
            unique_id_to_pos[unique_id] = index

        for feature in self.qa_features:
            example_index_to_features[feature.example_index].append(feature)

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction", [
                "feature_index", "start_index", "end_index", "start_logit",
                "end_logit"
            ])

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        scores_diff_json = collections.OrderedDict()

        for (example_index, example) in enumerate(self.qa_examples):

            features = example_index_to_features[example_index]

            prelim_predictions = []
            # keep track of the minimum score of null start+end of position 0
            score_null = 1000000  # large and positive
            min_null_feature_index = 0  # the paragraph slice with min null score
            null_start_logit = 0  # start logit at the slice with min null score
            null_end_logit = 0  # end logit at the slice with min null score
            for (feature_index, feature) in enumerate(features):
                pos = unique_id_to_pos[feature.unique_id]
                start_indexes = _get_best_indexes(start_logits[pos],
                                                  n_best_size)
                end_indexes = _get_best_indexes(end_logits[pos], n_best_size)
                # if we could have irrelevant answers,
                # get the min score of irrelevant
                if version_2_with_negative:
                    feature_null_score = start_logits[pos][0] + end_logits[
                        pos][0]
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
                        if not feature.token_is_max_context.get(
                                start_index, False):
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
                                end_logit=end_logits[pos][end_index]))

            if version_2_with_negative:
                prelim_predictions.append(
                    _PrelimPrediction(feature_index=min_null_feature_index,
                                      start_index=0,
                                      end_index=0,
                                      start_logit=null_start_logit,
                                      end_logit=null_end_logit))
            prelim_predictions = sorted(
                prelim_predictions,
                key=lambda x: (x.start_logit + x.end_logit),
                reverse=True)

            _NbestPrediction = collections.namedtuple(
                "NbestPrediction", ["text", "start_logit", "end_logit"])

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break
                feature = features[pred.feature_index]
                if pred.start_index > 0:  # this is a non-null prediction
                    tok_tokens = feature.tokens[pred.start_index:(
                        pred.end_index + 1)]
                    orig_doc_start = feature.token_to_orig_map[
                        pred.start_index]
                    orig_doc_end = feature.token_to_orig_map[pred.end_index]
                    orig_tokens = example.doc_tokens[orig_doc_start:(
                        orig_doc_end + 1)]
                    tok_text = " ".join(tok_tokens)

                    # De-tokenize WordPieces that have been split off.
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")

                    # Clean whitespace
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    orig_text = " ".join(orig_tokens)

                    final_text = get_final_text(tok_text, orig_text,
                                                do_lower_case)
                    if final_text in seen_predictions:
                        continue

                    seen_predictions[final_text] = True
                else:
                    final_text = ""
                    seen_predictions[final_text] = True

                nbest.append(
                    _NbestPrediction(text=final_text,
                                     start_logit=pred.start_logit,
                                     end_logit=pred.end_logit))
            # if we didn't include the empty option in the n-best, include it
            if version_2_with_negative:
                if "" not in seen_predictions:
                    nbest.append(
                        _NbestPrediction(text="",
                                         start_logit=null_start_logit,
                                         end_logit=null_end_logit))

                # In very rare edge cases we could only have single null pred.
                # We just create a nonce prediction in this case to avoid failure.
                if len(nbest) == 1:
                    nbest.insert(
                        0,
                        _NbestPrediction(text="empty",
                                         start_logit=0.0,
                                         end_logit=0.0))

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                    _NbestPrediction(text="empty",
                                     start_logit=0.0,
                                     end_logit=0.0))

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
                score_diff = score_null - best_non_null_entry.start_logit - (
                    best_non_null_entry.end_logit)
                scores_diff_json[example.qas_id] = score_diff
                if score_diff > null_score_diff_threshold:
                    all_predictions[example.qas_id] = ""
                else:
                    all_predictions[example.qas_id] = best_non_null_entry.text
                all_nbest_json[example.qas_id] = nbest_json

        # with open("output_predictions.json", "w") as writer:
        #    writer.write(json.dumps(all_predictions, indent=4) + "\n")

        return all_predictions, all_nbest_json, scores_diff_json

    def evaluate_predictions(self, all_predictions):

        exact_match = 0.
        f1 = 0.
        # Loop over all the examples and evaluate the predictions
        for example in self.qa_examples:

            qas_id = example.qas_id
            if qas_id not in all_predictions:
                continue

            ground_truths = example.ground_truths
            prediction = all_predictions[qas_id]

            exact_match += metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truths)

            f1 += metric_max_over_ground_truths(f1_score, prediction,
                                                ground_truths)

        exact_match = 100.0 * exact_match / len(self.qa_examples)
        f1 = 100.0 * f1 / len(self.qa_examples)

        return exact_match, f1

    def calculate_exact_match_and_f1(self, unique_ids, start_logits,
                                     end_logits, n_best_size,
                                     max_answer_length, do_lower_case,
                                     version_2_with_negative,
                                     null_score_diff_threshold):

        all_predictions, all_nbest_json, scores_diff_json = \
            self.get_predictions(unique_ids, start_logits, end_logits, n_best_size,
                                 max_answer_length, do_lower_case,
                                 version_2_with_negative, null_score_diff_threshold)

        exact_match, f1 = self.evaluate_predictions(all_predictions)

        return exact_match, f1


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 ground_truths,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.ground_truths = ground_truths
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):

        query_tokens = tokenizer.text_to_tokens(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.text_to_tokens(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position +
                                                     1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position,
                tokenizer, example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
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
                token_to_orig_map[len(tokens)] = \
                    tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans,
                                                       doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = 0
            end_position = 0
            if is_training and not example.is_impossible:
                # For training, if our document chunk does not contain an
                # annotation we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start
                        and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0

            if example_index < 1:
                print("*** Example ***")
                print("unique_id: %s" % unique_id)
                print("example_index: %s" % example_index)
                print("doc_span_index: %s" % doc_span_index)
                print("tokens: %s" % " ".join(tokens))
                print("token_to_orig_map: %s" % " ".join(
                    ["%d:%d" % (x, y)
                     for (x, y) in token_to_orig_map.items()]))
                print("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y)
                    for (x, y) in token_is_max_context.items()
                ]))
                print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                print("input_mask: %s" % " ".join([str(x)
                                                   for x in input_mask]))
                print("segment_ids: %s" %
                      " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    print("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(
                        tokens[start_position:(end_position + 1)])
                    print("start_position: %d" % start_position)
                    print("end_position: %d" % end_position)
                    print("answer: %s" % answer_text)

            features.append(
                InputFeatures(unique_id=unique_id,
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
                              is_impossible=example.is_impossible))

            unique_id += 1

    return features


def read_qa_examples(input_file, data_format, is_training):
    if data_format.lower() == "squad_json":
        examples = read_squad_examples(input_file, is_training)
    else:
        raise ValueError(f"Invalid format in QADataLayerForPretrainedModel: "
                         f"{data_format}")

    return examples


def read_squad_examples(input_file, is_training):
    version_2_with_negative = False
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = -1
                end_position = -1
                orig_answer_text = None
                is_impossible = False
                ground_truths = []
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have "
                            "exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset +
                                                           answer_length - 1]
                        # Only add answers where the text can be exactly
                        # recovered from the document. If this CAN'T happen
                        # it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example
                        # is NOT guaranteed to be preserved.
                        actual_text = " ".join(
                            doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            print("Could not find answer: '%s' vs. '%s'",
                                  actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""
                else:  # Eval data set
                    # Store the potential answers in examples
                    ground_truths = list(
                        map(lambda x: x['text'], qa['answers']))

                example = SquadExample(qas_id=qas_id,
                                       question_text=question_text,
                                       ground_truths=ground_truths,
                                       doc_tokens=doc_tokens,
                                       orig_answer_text=orig_answer_text,
                                       start_position=start_position,
                                       end_position=end_position,
                                       is_impossible=is_impossible)
                examples.append(example)

    return examples


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the
    annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we
    # can often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can
    # match the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.text_to_tokens(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & \
             collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to
    # the span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic
    # between `pred_text` and `orig_text` to get a character-to-character
    # alignment. This can fail in certain cases in which case we just return
    # `orig_text`.

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
            print("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            print("Length not equal after stripping spaces: '%s' vs '%s'",
                  orig_ns_text, tok_ns_text)
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
            print("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            print("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text
