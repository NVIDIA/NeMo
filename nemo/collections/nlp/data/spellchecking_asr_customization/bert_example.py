# Copyright 2019 The Google Research Authors.
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import logging
from collections import OrderedDict
from os import path
from typing import Dict, List, Optional, Tuple, Union

from transformers import PreTrainedTokenizerBase

import pdb

"""Build BERT Examples from asr hypothesis, customization candidate, target labels, span info.
"""

class BertExample(object):
    """Class for training and inference examples for BERT.

    Attributes:
        features: Feature dictionary.
    """

    def __init__(
        self,
        input_ids: List[int],
        input_mask: List[int],
        segment_ids: List[int],
        labels_mask: List[int],
        labels: List[int],
        spans: List[Tuple[int, int, int]],
        default_label: int,
    ) -> None:
        """Inputs to the example wrapper

        Args:
            input_ids: indices of tokens which constitute batches of masked text segments
            input_mask: bool tensor with 0s in place of source tokens to be masked
            segment_ids: bool tensor with 0's and 1's to denote the text segment type
            labels_mask: bool tensor with 0s in place of label tokens to be masked
            labels: indices of semiotic classes which should be predicted from each of the
                corresponding input tokens
            spans: list of tuples (class_id, start_wordpiece_idx, end_wordpiece_idx), end is exclusive
            default_label: The default label
        """
        input_len = len(input_ids)
        if not (
            input_len == len(input_mask)
            and input_len == len(segment_ids)
            and input_len == len(labels_mask)
            and input_len == len(labels)
        ):
            raise ValueError('All feature lists should have the same length ({})'.format(input_len))

        self.features = OrderedDict(
            [
                ("input_ids", input_ids),
                ("input_mask", input_mask),
                ("segment_ids", segment_ids),
                ("labels_mask", labels_mask),
                ("labels", labels),
                ("spans", spans),
            ]
        )
        self._default_label = default_label

    def pad_to_max_length(self, max_seq_length: int, max_spans_length: int, pad_token_id: int) -> None:
        """Pad the feature vectors so that they all have max_seq_length.

        Args:
            max_seq_length: The length that all features, except semiotic_classes, will have after padding.
            max_spans_length: The length that spans will have after padding.
            pad_token_id: input_ids feature is padded with this ID, other features
                with ID 0.
        """
        pad_len = max_seq_length - len(self.features['input_ids'])
        self.features["spans"].extend(
            [(-1, -1, -1)] * (max_spans_length - len(self.features["spans"]))
        )
        for key in self.features:
            if key == "spans":
                continue
            pad_id = pad_token_id if (key == "input_ids") else 0
            self.features[key].extend([pad_id] * pad_len)
            if len(self.features[key]) != max_seq_length:
                raise ValueError(
                    "{} has length {} (should be {}).".format(key, len(self.features[key]), max_seq_length)
                )

    def get_token_labels(self, features_key: str) -> List[int]:
        """Returns labels/tags for the original tokens, not for wordpieces."""
        labels = []
        for idx in range(len(self.features[features_key])):
            # For unmasked and untruncated tokens, use the label in the features, and
            # for the truncated tokens, use the default label.
            if idx < len(self.features[features_key]) and self.features["labels_mask"][idx]:
                labels.append(self.features[features_key][idx])
            else:
                labels.append(self._default_label)
        return labels


class BertExampleBuilder(object):
    """Builder class for BertExample objects."""

    def __init__(
        self,
        label_map: Dict[str, int],
        semiotic_classes: Dict[str, int],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
    ) -> None:
        """Initializes an instance of BertExampleBuilder.

        Args:
            label_map: Mapping from tags to tag IDs.
            semiotic_classes: Mapping from semiotic classes to their ids.
            tokenizer: Tokenizer object.
            max_seq_length: Maximum sequence length.
        """
        self._label_map = label_map
        self._semiotic_classes = semiotic_classes
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._max_spans_length = max(4, int(max_seq_length / 2))
        self._pad_id = self._tokenizer.pad_token_id
        self._keep_tag_id = 0

    def build_bert_example_simple(
        self, hyp: str, ref: str, target: Optional[str] = None, span_info: Optional[str] = None, infer: bool = False
    ) -> Optional[BertExample]:
        """Constructs a BERT Example.

        Args:
            hyp: Hypothesis text.
            ref: Candidate customization variant
            target: String of labels or None when building an example during inference.
            span_info: String or None
            infer: inference mode
        Returns:
            BertExample, or None if the conversion from text to tags was infeasible
        """
        # Compute target labels.
        if (target is not None) and (not infer):
            tags = list(map(int, target.split()))
            if not tags:
                return None
        else:
            # If target is not provided, we set all target labels to 0.
            tags = [0 for _ in hyp.split()]
        hyp_tokens, labels, token_start_indices = self._split_to_wordpieces_with_labels(hyp.split(), tags)
        ref_tokens, ref_start_indices = self._split_to_wordpieces(ref.split())

        input_tokens = ["[CLS]"] + hyp_tokens + ["[SEP]"] + ref_tokens + ["[SEP]"]
        labels_mask = [0] + [1] * len(labels) + [0] + [0] * len(ref_tokens) + [0]
        labels = [0] + labels + [0] + [0] * len(ref_tokens) + [0]

        input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] + [0] * len(hyp_tokens) + [0] + [1] * len(ref_tokens) + [1]

        if len(input_ids) != len(segment_ids):
            raise ValueError(
                "len(input_ids)=" + str(len(input_ids)) + " is different from len(segment_ids)=" + str(len(segment_ids))
            )

        if "PLAIN" not in self._semiotic_classes:
            raise KeyError("PLAIN should be in self._semiotic_classes")
        plain_cid = self._semiotic_classes["PLAIN"]
        semiotic_labels = [plain_cid] * len(labels)  # we use the same mask for semiotic labels as for tag labels

        spans = []

        if span_info is not None:
            # e.g. span_info="CUSTOM 0 5;CUSTOM 9 12"
            # translate class name to its id, translate coords from tokens to wordpieces
            span_info_parts = span_info.split(";")
            previous_end = 0
            for p in span_info_parts:
                if p == "":
                    break
                c, start, end = p.split(" ")
                if c not in self._semiotic_classes:
                    raise KeyError("c=" + c + " not found in self._semiotic_classes")
                cid = self._semiotic_classes[c]
                start = int(start)
                end = int(end)
                if start >= len(token_start_indices):
                    raise IndexError(
                        "start=" + str(start) + " is outside len(token_start_indices)=" + str(len(token_start_indices))
                    )
                while previous_end < start:
                    subtoken_start = token_start_indices[previous_end]
                    subtoken_end = (
                        token_start_indices[previous_end + 1]
                        if previous_end + 1 < len(token_start_indices)
                        else len(input_ids) - 1
                    )
                    spans.append((plain_cid, subtoken_start, subtoken_end))
                    previous_end += 1
                subtoken_start = token_start_indices[start]
                subtoken_end = token_start_indices[end] if end < len(token_start_indices) else len(hyp_tokens) + 1
                if subtoken_end >= self._max_seq_length:  # possible if input_ids gets truncated to the max_seq_length
                    break
                spans.append((cid, subtoken_start, subtoken_end))
                previous_end = end
            while previous_end < len(token_start_indices):
                subtoken_start = token_start_indices[previous_end]
                subtoken_end = (
                    token_start_indices[previous_end + 1]
                    if previous_end + 1 < len(token_start_indices)
                    else len(input_ids) - 1
                )
                spans.append((plain_cid, subtoken_start, subtoken_end))
                previous_end += 1
        if len(input_ids) > self._max_seq_length or len(spans) > self._max_spans_length:
            return None
        example = BertExample(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            labels_mask=labels_mask,
            labels=labels,
            spans=spans,
            default_label=0,
        )
        example.pad_to_max_length(self._max_seq_length, self._max_spans_length, self._pad_id)
        #pdb.set_trace()
        return example

    def build_bert_example(
        self, hyp: str, ref: str, target: Optional[str] = None, span_info: Optional[str] = None, infer: bool = False
    ) -> Optional[BertExample]:
        """Constructs a BERT Example.

        Args:
            hyp: Hypothesis text.
            ref: Candidate customization variants divided by ';'
            target: String of labels(1-based index of correct example or 0) or None when building an example during inference.
            span_info: String of format "CUSTOM 6 20;CUSTOM 40 51", number of parts corresponds to number of targets
            infer: inference mode
        Returns:
            BertExample, or None if the conversion from text to tags was infeasible
        """
        tags = [0 for _ in hyp.split()]
        if target is not None:
            for p, t in zip(span_info.split(";"), target.split(" ")):
                c, start, end = p.split(" ")
                start = int(start)
                end = int(end)
                tags[start:end] = [int(t) for i in range(end-start)]

        hyp_tokens, labels, token_start_indices = self._split_to_wordpieces_with_labels(hyp.split(), tags)
        references = ref.split(";")
        all_ref_tokens = []
        all_ref_segment_ids = []
        for i in range(len(references)):
            ref_tokens, _ = self._split_to_wordpieces(references[i].split())
            all_ref_tokens.extend(ref_tokens + ["[SEP]"])
            all_ref_segment_ids.extend([i + 1] * (len(ref_tokens) + 1))

        input_tokens = ["[CLS]"] + hyp_tokens + ["[SEP]"] + all_ref_tokens   # ends with [SEP]
        labels_mask = [0] + [1] * len(labels) + [0] + [0] * len(all_ref_tokens)
        labels = [0] + labels + [0] + [0] * len(all_ref_tokens)

        input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] + [0] * len(hyp_tokens) + [0] + all_ref_segment_ids

        if len(input_ids) != len(segment_ids):
            raise ValueError(
                "len(input_ids)=" + str(len(input_ids)) + " is different from len(segment_ids)=" + str(len(segment_ids))
            )

        if "PLAIN" not in self._semiotic_classes:
            raise KeyError("PLAIN should be in self._semiotic_classes")
        plain_cid = self._semiotic_classes["PLAIN"]

        spans = []

        if span_info is not None:
            # e.g. span_info="CUSTOM 0 5;CUSTOM 9 12"
            # translate class name to its id, translate coords from tokens to wordpieces
            span_info_parts = span_info.split(";")
            previous_end = 0
            for p in span_info_parts:
                if p == "":
                    break
                c, start, end = p.split(" ")
                if c not in self._semiotic_classes:
                    raise KeyError("c=" + c + " not found in self._semiotic_classes")
                cid = self._semiotic_classes[c]
                start = int(start)
                end = int(end)
                if start >= len(token_start_indices):
                    raise IndexError(
                        "start=" + str(start) + " is outside len(token_start_indices)=" + str(len(token_start_indices))
                    )
                while previous_end < start:
                    subtoken_start = token_start_indices[previous_end]
                    subtoken_end = (
                        token_start_indices[previous_end + 1]
                        if previous_end + 1 < len(token_start_indices)
                        else len(input_ids) - 1
                    )
                    spans.append((plain_cid, subtoken_start, subtoken_end))
                    previous_end += 1
                subtoken_start = token_start_indices[start]
                subtoken_end = token_start_indices[end] if end < len(token_start_indices) else len(hyp_tokens) + 1
                if subtoken_end >= self._max_seq_length:  # possible if input_ids gets truncated to the max_seq_length
                    break
                spans.append((cid, subtoken_start, subtoken_end))
                previous_end = end
            while previous_end < len(token_start_indices):
                subtoken_start = token_start_indices[previous_end]
                subtoken_end = (
                    token_start_indices[previous_end + 1]
                    if previous_end + 1 < len(token_start_indices)
                    else len(input_ids) - 1
                )
                spans.append((plain_cid, subtoken_start, subtoken_end))
                previous_end += 1
        if len(input_ids) > self._max_seq_length or len(spans) > self._max_spans_length:
            return None
        example = BertExample(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            labels_mask=labels_mask,
            labels=labels,
            spans=spans,
            default_label=0,
        )
        return example


    def _split_to_wordpieces_with_labels(self, tokens: List[str], labels: List[int]) -> Tuple[List[str], List[int], List[int]]:
        """Splits tokens (and the labels accordingly) to WordPieces.

        Args:
            tokens: Tokens to be split.
            labels: Labels (one per token) to be split.

        Returns:
            3-tuple with the split tokens, split labels, and the indices of the
            WordPieces that start a token.
        """
        bert_tokens = []  # Original tokens split into wordpieces.
        bert_labels = []  # Label for each wordpiece.
        # Index of each wordpiece that starts a new token.
        token_start_indices = []
        for i, token in enumerate(tokens):
            # '+ 1' is because bert_tokens will be prepended by [CLS] token later.
            token_start_indices.append(len(bert_tokens) + 1)
            pieces = self._tokenizer.tokenize(token)
            bert_tokens.extend(pieces)
            bert_labels.extend([labels[i]] * len(pieces))
        return bert_tokens, bert_labels, token_start_indices

    def _split_to_wordpieces(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        """Splits tokens to WordPieces.

        Args:
            tokens: Tokens to be split.

        Returns:
            tuple with the split tokens, and the indices of the WordPieces that start a token.
        """
        bert_tokens = []  # Original tokens split into wordpieces.
        # Index of each wordpiece that starts a new token.
        token_start_indices = []
        for i, token in enumerate(tokens):
            # '+ 1' is because bert_tokens will be prepended by [CLS] token later.
            token_start_indices.append(len(bert_tokens) + 1)
            pieces = self._tokenizer.tokenize(token)
            bert_tokens.extend(pieces)
        return bert_tokens, token_start_indices

    def _get_pad_id(self) -> int:
        """Returns the ID of the [PAD] token (or 0 if it's not in the vocab)."""
        try:
            return self._tokenizer.pad_token_id
        except KeyError:
            return 0


def read_input_file(
    example_builder: 'BertExampleBuilder', input_filename: str, infer: bool = False
) -> List['BertExample']:
    """Reads in Tab Separated Value file and converts to training/inference-ready examples.

    Args:
        example_builder: Instance of BertExampleBuilder
        input_filename: Path to the TSV input file.
        infer: Whether test files or not.

    Returns:
        examples: List of converted examples(features and Editing Tasks)
    """

    if not path.exists(input_filename):
        raise ValueError("Cannot find file: " + input_filename)
    examples = []
    with open(input_filename, 'r') as f:
        for line in f:
            if len(examples) % 1000 == 0:
                logging.info("{} examples processed.".format(len(examples)))
            hyp, ref, target, semiotic_info = line.rstrip('\n').split('\t')
            example = example_builder.build_bert_example(hyp, ref, target, semiotic_info, infer)
            if example is None:
                continue
            examples.append(example)
    logging.info(f"Done. {len(examples)} examples converted.")
    return examples
