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
Utility functions for Token Classification NLP tasks
Some transformer of this code were adapted from the HuggingFace library at
https://github.com/huggingface/pytorch-pretrained-BERT
"""

#TODO: REFACTOR to minimize code reusing


import collections
import numpy as np
from torch.utils.data import Dataset
import string
import re
import random
from ...externals.run_squad import _check_is_max_context


def remove_punctuation_from_sentence(sentence):
    sentence = re.sub('[' + string.punctuation + ']', '', sentence)
    sentence = sentence.lower()
    return sentence


class BertTokenClassificationDataset(Dataset):
    def __init__(self, input_file, max_seq_length, tokenizer):

        # Read the sentences and group them in sequences up to max_seq_length
        with open(input_file, "r") as f:
            self.seq_words = []
            self.seq_token_labels = []
            self.seq_sentence_labels = []
            self.seq_subtokens = []

            new_seq_words = []
            new_seq_token_labels = []
            new_seq_sentence_labels = []
            new_seq_subtokens = []
            new_seq_subtoken_count = 0

            lines = f.readlines()
            random.seed(0)
            random.shuffle(lines)

            for index, line in enumerate(lines):

                if index % 20000 == 0:
                    print(f"processing line {index}/{len(lines)}")

                sentence_label = line.split()[0]
                sentence = line.split()[2:]
                sentence = " ".join(sentence)
                # Remove punctuation
                sentence = remove_punctuation_from_sentence(sentence)
                sentence_words = sentence.split()

                sentence_subtoken_count = 0
                sentence_subtokens = []
                for word in sentence_words:
                    word_tokens = tokenizer.text_to_tokens(word)
                    sentence_subtokens.append(word_tokens)
                    sentence_subtoken_count += len(word_tokens)

                sentence_token_labels = [0] * sentence_subtoken_count
                sentence_token_labels[0] = 1

                # The -1 accounts for [CLS]
                max_tokens_for_doc = max_seq_length - 1

                if (new_seq_subtoken_count + sentence_subtoken_count) < \
                    max_tokens_for_doc:

                    new_seq_words.extend(sentence_words)
                    new_seq_token_labels.extend(sentence_token_labels)
                    new_seq_sentence_labels.append(sentence_label)
                    new_seq_subtokens.append(sentence_subtokens)
                    new_seq_subtoken_count += sentence_subtoken_count

                else:
                    self.seq_words.append(new_seq_words)
                    self.seq_token_labels.append(new_seq_token_labels)
                    self.seq_sentence_labels.append(new_seq_sentence_labels)
                    self.seq_subtokens.append(new_seq_subtokens)

                    new_seq_words = sentence_words
                    new_seq_token_labels = sentence_token_labels
                    new_seq_sentence_labels = [sentence_label]
                    new_seq_subtokens = [sentence_subtokens]
                    new_seq_subtoken_count = sentence_subtoken_count

        self.features = convert_sequences_to_features(
            self.seq_words, self.seq_subtokens, self.seq_token_labels,
            self.seq_sentence_labels, tokenizer, max_seq_length)

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.vocab_size = self.tokenizer.vocab_size

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        feature = self.features[idx]

        return np.array(feature.input_ids), np.array(feature.segment_ids), \
            np.array(feature.input_mask, dtype=np.float32)[..., None], \
            np.array(feature.labels), np.array(feature.seq_id)

    def eval_preds(self, logits_lists, seq_ids):

        # Count the number of correct and incorrect predictions
        correct_labels = 0
        incorrect_labels = 0

        correct_preds = 0
        total_preds = 0
        total_correct = 0

        for logits, seq_id in zip(logits_lists, seq_ids):

            feature = self.features[seq_id]

            masks = feature.input_mask
            last_mask_index = masks.index(0)
            labels = feature.labels[:last_mask_index]
            labels = labels[:last_mask_index]
            logits = logits[:last_mask_index]

            preds = [1 if (a[1] > a[0]) else 0 for a in logits]

            correct_preds = 0
            correct_labels = 0
            for label, pred in zip(labels, preds):
                if pred == label:
                    correct_labels += 1
                    if pred == 1:
                        correct_preds += 1

            total_preds = preds.count(1)
            total_correct = labels.count(1)
            incorrect_labels = len(labels) - correct_labels

            if seq_id < 1:
                previous_word_id = -1
                predicted_seq = ""
                correct_seq = ""
                unpunctuated_seq = ""

                for token_id, word_id in feature.token_to_orig_map.items():

                    word = feature.words[word_id]

                    if word_id is not previous_word_id:
                        # New words has been found, handle it
                        if feature.labels[token_id] is 1:
                            if previous_word_id is not -1:
                                correct_seq += ". "
                            correct_seq += word.capitalize()
                        else:
                            correct_seq += " " + word

                        if preds[token_id] is 1:
                            if previous_word_id is not -1:
                                predicted_seq += ". "
                            predicted_seq += word.capitalize()
                        else:
                            predicted_seq += " " + word

                        unpunctuated_seq += " " + word

                    previous_word_id = word_id

                print("unpunctuated_seq:\n", unpunctuated_seq)
                print("correct_seq:\n", correct_seq)
                print("pred_seq:\n", predicted_seq)

        return correct_labels, incorrect_labels, correct_preds, total_preds, \
            total_correct


def convert_sequences_to_features(seqs_words, seqs_subtokens,
                                  seqs_token_labels, seqs_sentence_labels,
                                  tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for seq_id, (words, seq_subtokens, seq_token_labels, sentence_labels) in \
        enumerate(zip(seqs_words, seqs_subtokens, seqs_token_labels,
            seqs_sentence_labels)):

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []

        word_count = 0
        for sent_subtokens in seq_subtokens:
            for word_subtokens in sent_subtokens:
                orig_to_tok_index.append(len(all_doc_tokens))
                for sub_token in word_subtokens:
                    tok_to_orig_index.append(word_count)
                    all_doc_tokens.append(sub_token)
                word_count += 1

        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        length = len(all_doc_tokens)
        doc_spans.append(_DocSpan(start=start_offset, length=length))

        doc_span_index = 0
        doc_span = doc_spans[0]

        tokens = []
        token_labels = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        token_labels.append(0)
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(
                tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(0)

        for label in seq_token_labels:
            token_labels.append(label)

        input_ids = tokenizer.tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            token_labels.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(token_labels) == max_seq_length

        if seq_id < 1:
            print("*** Example ***")
            print("example_index: %s" % seq_id)
            print("doc_span_index: %s" % doc_span_index)
            print("tokens: %s" % " ".join(tokens))
            print("words: %s" % " ".join(words))
            print("labels: %s" % " ".join(str(token_labels)))
            print("sentence_labels: %s" % " ".join(sentence_labels))
            print("token_to_orig_map: %s" % " ".join(
                ["%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
            print("token_is_max_context: %s" % " ".join(
                ["%d:%s" % (x, y) for (x, y) in token_is_max_context.items()]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(seq_id=seq_id,
                          doc_span_index=doc_span_index,
                          tokens=tokens,
                          words=words,
                          labels=token_labels,
                          sentence_labels=sentence_labels,
                          token_to_orig_map=token_to_orig_map,
                          token_is_max_context=token_is_max_context,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids))

    return features


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, seq_id, doc_span_index, tokens, words, labels,
                 sentence_labels, token_to_orig_map, token_is_max_context,
                 input_ids, input_mask, segment_ids):
        self.seq_id = seq_id
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.words = words
        self.labels = labels
        self.sentence_labels = sentence_labels
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
