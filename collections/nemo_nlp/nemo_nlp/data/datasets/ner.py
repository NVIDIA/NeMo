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
Utility functions for NER NLP tasks
Some transformer of this code were adapted from the HuggingFace library at
https://github.com/huggingface/pytorch-pretrained-BERT
"""

# TODO: REFACTOR to minimize code reusing

import collections
import numpy as np
from torch.utils.data import Dataset


class BertNERDataset(Dataset):
    def __init__(self, input_file, max_seq_length, tokenizer):
        # Read the sentences and group them in sequences up to max_seq_length
        with open(input_file, "r") as f:
            self.seq_words = []
            self.seq_token_labels = []
            self.seq_subtokens = []

            new_seq_words = []
            new_seq_token_labels = []
            new_seq_subtokens = []
            new_seq_subtoken_count = 0

            lines = f.readlines()

            words = []
            tags = []
            tokens = []
            token_tags = []
            token_count = 0

            def process_sentence():
                nonlocal new_seq_words, new_seq_token_labels, \
                    new_seq_subtokens, new_seq_subtoken_count

                # The -1 accounts for [CLS]
                max_tokens_for_doc = max_seq_length - 1

                if (new_seq_subtoken_count + token_count) < max_tokens_for_doc:
                    new_seq_words.extend(words)
                    new_seq_token_labels.extend(token_tags)
                    new_seq_subtokens.append(tokens)
                    new_seq_subtoken_count += token_count
                else:
                    self.seq_words.append(new_seq_words)
                    self.seq_token_labels.append(new_seq_token_labels)
                    self.seq_subtokens.append(new_seq_subtokens)

                    new_seq_words = words
                    new_seq_token_labels = token_tags
                    new_seq_subtokens = [tokens]
                    new_seq_subtoken_count = token_count

            all_tags = {}

            # Collect a list of all possible tags
            for line in lines:
                if line == "\n":
                    continue

                tag = line.split()[-1]

                if tag not in tags and tag != "O":
                    tag = tag.split("-")[1]
                    all_tags[tag] = 0

            # Create mapping of tags to tag ids that starts with "O" -> 0 and
            # then increases in alphabetical order
            tag_ids = {"O": 0}

            for tag in sorted(all_tags):
                tag_ids[tag] = len(tag_ids)

            # Process all lines in input data
            for line in lines:
                if line == "\n":
                    # If we hit a newline, we've reached the end of a sentence
                    process_sentence()
                    words = []
                    tags = []
                    tokens = []
                    token_tags = []
                    continue

                word = line.split()[0]
                tag = line.split()[-1]

                if tag != "O":
                    tag = tag.split("-")[1]

                word_tokens = tokenizer.text_to_tokens(word)
                tag_id = tag_ids[tag]

                words.append(word)
                tags.append(tag_id)
                tokens.append(word_tokens)
                token_tags.extend([tag_id] * len(word_tokens))
                token_count += len(word_tokens)

            self.features = convert_sequences_to_features(
                self.seq_words, self.seq_subtokens, self.seq_token_labels,
                tokenizer, max_seq_length)

            self.tag_ids = tag_ids
            self.tokenizer = tokenizer
            self.max_seq_length = max_seq_length
            self.vocab_size = self.tokenizer.vocab_size

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]

        return np.array(feature.input_ids), np.array(feature.segment_ids), \
            np.array(feature.input_mask, dtype=np.float32), \
            np.array(feature.labels), np.array(feature.seq_id)

    def eval_preds(self, logits_lists, seq_ids, tag_ids):
        correct_chunks = 0
        total_chunks = 0
        predicted_chunks = 0

        correct_tags = 0
        token_count = 0

        lines = []
        tag_ids = {tag_ids[k]: k for k in tag_ids}

        for logits, seq_id in zip(logits_lists, seq_ids):
            feature = self.features[seq_id]
            masks = feature.input_mask

            try:
                last_mask_index = masks.index(0)
            except ValueError:
                last_mask_index = len(masks)

            labels = feature.labels[:last_mask_index]
            labels = np.array(labels[:last_mask_index])
            logits = logits[:last_mask_index]
            preds = np.argmax(logits, axis=1)

            in_correct = False

            last_label = "O"
            last_pred = "O"

            last_correct = "O"
            last_correct_type = ""
            last_guessed = "O"
            last_guessed_type = ""

            previous_word_id = -1

            # Adapted from conlleval.pl (included with CoNLL-2003 dataset)
            for token_id, word_id in feature.token_to_orig_map.items():
                if word_id is not previous_word_id:
                    word = feature.words[word_id]
                    label = tag_ids[feature.labels[token_id]]
                    pred = tag_ids[preds[token_id]]

                    if pred != "O":
                        guessed = "B" if last_pred != pred else "I"
                        guessed_type = pred
                    else:
                        guessed = pred
                        guessed_type = ""

                    if label != "O":
                        correct = "B" if last_label != label else "I"
                        correct_type = label
                    else:
                        correct = label
                        correct_type = ""

                    if in_correct:
                        if end_of_chunk(last_correct, correct,
                                        last_correct_type, correct_type) and \
                                end_of_chunk(last_guessed, guessed,
                                             last_guessed_type,
                                             guessed_type) and \
                                last_guessed_type == last_correct_type:
                            in_correct = False
                            correct_chunks += 1

                        elif end_of_chunk(last_correct, correct,
                                          last_correct_type, correct_type) != \
                                end_of_chunk(last_guessed, guessed,
                                             last_guessed_type,
                                             guessed_type) or \
                                guessed_type != correct_type:
                            in_correct = False

                    if start_of_chunk(last_correct, correct, last_correct_type,
                                      correct_type) and \
                            start_of_chunk(last_guessed, guessed,
                                           last_guessed_type,
                                           guessed_type) and \
                            guessed_type == correct_type:
                        in_correct = True

                    if start_of_chunk(last_correct, correct, last_correct_type,
                                      correct_type):
                        total_chunks += 1

                    if start_of_chunk(last_guessed, guessed, last_guessed_type,
                                      guessed_type):
                        predicted_chunks += 1

                    if correct == guessed and guessed_type == correct_type:
                        correct_tags += 1

                    token_count += 1

                    last_label = label
                    last_pred = pred

                    last_guessed = guessed
                    last_correct = correct
                    last_guessed_type = guessed_type
                    last_correct_type = correct_type

                    lines.append({
                        "word": word,
                        "label": feature.labels[token_id],
                        "prediction": preds[token_id]
                    })

                previous_word_id = word_id

            if in_correct:
                correct_chunks += 1

            lines.append({
                "word": ""
            })

        return correct_tags, token_count, correct_chunks, predicted_chunks, \
            total_chunks, lines


def start_of_chunk(prev_tag, tag, prev_type, type):
    if prev_tag == "B" and tag == "B":
        return True
    elif prev_tag == "I" and tag == "B":
        return True
    elif prev_tag == "O" and tag == "B":
        return True
    elif prev_tag == "O" and tag == "I":
        return True
    elif prev_tag == "O" and tag == "I":
        return True
    elif tag != "O" and prev_type != type:
        return True

    return False


def end_of_chunk(prev_tag, tag, prev_type, type):
    if prev_tag == "B" and tag == "B":
        return True
    elif prev_tag == "B" and tag == "O":
        return True
    elif prev_tag == "I" and tag == "B":
        return True
    elif prev_tag == "I" and tag == "O":
        return True
    elif prev_tag == "I" and tag == "O":
        return True
    elif prev_tag != "O" and prev_type != type:
        return True

    return False


def convert_sequences_to_features(seqs_words, seqs_subtokens,
                                  seqs_token_labels, tokenizer,
                                  max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for seq_id, (words, seq_subtokens, seq_token_labels) in \
            enumerate(zip(seqs_words, seqs_subtokens, seqs_token_labels)):

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

        # Ensure that we don't go over the maximum sequence length
        for i in range(min(doc_span.length, max_seq_length - 1)):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = \
                tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(0)

        for label in seq_token_labels:
            if len(token_labels) == len(tokens):
                break

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

        features.append(
            InputFeatures(
                seq_id=seq_id,
                doc_span_index=doc_span_index,
                tokens=tokens,
                words=words,
                labels=token_labels,
                token_to_orig_map=token_to_orig_map,
                token_is_max_context=token_is_max_context,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids))

    return features


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a
    # single token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left
    # context and 0 right context.
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
        score = min(num_left_context, num_right_context) + \
            0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 seq_id,
                 doc_span_index,
                 tokens,
                 words,
                 labels,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids):
        self.seq_id = seq_id
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.words = words
        self.labels = labels
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
