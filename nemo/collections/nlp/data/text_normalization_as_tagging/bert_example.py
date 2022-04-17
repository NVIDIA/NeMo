# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (c) 2022 NVIDIA. All Rights Reserved.
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
# -*- coding: utf-8 -*-
#
# Copyright 2019 The Google Research Authors.
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
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/lasertagger/blob/master/bert_example.py
"""

from nemo.collections.nlp.data.text_normalization_as_tagging.tagging import Tag, EditingTask, TaggingConverterTrivial
from transformers import PreTrainedTokenizerBase
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional, Union

"""Build BERT Examples from source, target pairs.
   The difference from the original Lasertagger approach is that our target already consists of tags,
    so the preprocesssing is trivial.
"""


class BertExample(object):
    """Class for training and inference examples for BERT.

    Attributes:
        editing_task: The EditingTask from which this example was created. Needed
            when realizing labels predicted for this example.
        features: Feature dictionary.
    """

    def __init__(
        self,
        input_ids: List[int],
        input_mask: List[int],
        segment_ids: List[int],
        labels: List[int],
        labels_mask: List[int],
        token_start_indices: List[int],
        semiotic_classes: List[Tuple[int, int, int]],
        task: EditingTask,
        default_label: int,
    ) -> None:
        """Inputs to the example wrapper

        Args:
            input_ids: indices of tokens which constitute batches of masked text segments
            input_mask: bool tensor with 0s in place of source tokens to be masked
            segment_ids: bool tensor with 0's and 1's to denote the text segment type
            labels: indices of tokens which should be predicted from each of the
                corresponding input tokens
            labels_mask: bool tensor with 0s in place of label tokens to be masked
            token_start_indices: the indices of the WordPieces that start a token.
            semiotic_classes: list of tuples (class_id, start_wordpiece_idx, end_wordpiece_idx), end is exclusive
            task: Example Text-Editing Task used by the LaserTagger model during inference.
            default_label: The default label for the KEEP tag-ID
        """
        input_len = len(input_ids)
        if not (
            input_len == len(input_mask)
            and input_len == len(segment_ids)
            and input_len == len(labels)
            and input_len == len(labels_mask)
        ):
            raise ValueError('All feature lists should have the same length ({})'.format(input_len))

        self.features = OrderedDict(
            [
                ('input_ids', input_ids),
                ('input_mask', input_mask),
                ('segment_ids', segment_ids),
                ('labels', labels),
                ('labels_mask', labels_mask),
                ('semiotic_classes', semiotic_classes)
            ]
        )
        self._token_start_indices = token_start_indices
        self.editing_task = task
        self._default_label = default_label

    def pad_to_max_length(self, max_seq_length: int, max_semiotic_length: int, pad_token_id: int) -> None:
        """Pad the feature vectors so that they all have max_seq_length.

        Args:
            max_seq_length: The length that all features, except semiotic_classes, will have after padding.
            max_semiotic_length: The length that semiotic_classes will have after padding.
            pad_token_id: input_ids feature is padded with this ID, other features
                with ID 0.
        """
        pad_len = max_seq_length - len(self.features['input_ids'])
        self.features["semiotic_classes"].extend(
            [(-1, -1, -1)] * (max_semiotic_length - len(self.features['semiotic_classes'])))
        for key in self.features:
            if key == "semiotic_classes":
                continue
            pad_id = pad_token_id if (key == 'input_ids') else 0
            self.features[key].extend([pad_id] * pad_len)
            if len(self.features[key]) != max_seq_length:
                raise ValueError(
                    '{} has length {} (should be {}).'.format(key, len(self.features[key]), max_seq_length)
                )

    def get_token_labels(self) -> List[int]:
        """Returns labels/tags for the original tokens, not for wordpieces."""
        labels = []
        for idx in self._token_start_indices:
            # For unmasked and untruncated tokens, use the label in the features, and
            # for the truncated tokens, use the default label.
            if idx < len(self.features['labels']) and self.features['labels_mask'][idx]:
                labels.append(self.features['labels'][idx])
            else:
                labels.append(self._default_label)
        return labels


class BertExampleBuilder(object):
    """Builder class for BertExample objects."""

    def __init__(self,
                 label_map: Dict[str, int],
                 semiotic_classes: Dict[str, int],
                 tokenizer: PreTrainedTokenizerBase,
                 max_seq_length: int,
                 converter: TaggingConverterTrivial
                 ) -> None:
        """Initializes an instance of BertExampleBuilder.

        Args:
            label_map: Mapping from tags to tag IDs.
            semiotic_classes: Mapping from semiotic classes to their ids.
            tokenizer: Tokenizer object.
            max_seq_length: Maximum sequence length.
            converter: Converter from text targets to tags.
        """
        self._label_map = label_map
        self._semiotic_classes = semiotic_classes
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._max_semiotic_length = max(4, int(max_seq_length / 2))
        self._converter = converter
        self._pad_id = self._tokenizer.pad_token_id
        self._keep_tag_id = self._label_map['KEEP']

    def build_bert_example(
        self,
        source: str,
        target: Optional[str] = None,
        semiotic_info: Optional[str] = None,
        infer: bool = False,
    ) -> Optional[BertExample]:
        """Constructs a BERT Example.

        Args:
            source: Source text.
            target: Target text or None when building an example during inference.
            semiotic_info: String or None
            infer: inference mode
        Returns:
            BertExample, or None if the conversion from text to tags was infeasible
        """
        # Compute target labels.
        task = EditingTask(source)
        if (target is not None) and (not infer):
            tags = self._converter.compute_tags(task, target)
            if not tags:
                return None
        else:
            # If target is not provided, we set all target labels to KEEP.
            tags = [Tag('KEEP') for _ in task.source_tokens]
        labels = [self._label_map[str(tag)] for tag in tags]
        tokens, labels, token_start_indices = self._split_to_wordpieces(task.source_tokens, labels)

        tokens = self._truncate_list(tokens)
        labels = self._truncate_list(labels)

        input_tokens = ['[CLS]'] + tokens + ['[SEP]']
        labels_mask = [0] + [1] * len(labels) + [0]
        labels = [0] + labels + [0]

        input_ids = self._tokenizer.convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        semiotic_classes = []

        if semiotic_info is not None:

            # translate class name to its id, translate coords from tokens to wordpieces
            semiotic_info_parts = semiotic_info.split(";")
            previous_end = 0
            assert("PLAIN" in self._semiotic_classes)
            plain_cid = self._semiotic_classes["PLAIN"]
            for p in semiotic_info_parts:
                if p == "":
                    break
                c, start, end = p.split(" ")
                assert(c in self._semiotic_classes)
                cid = self._semiotic_classes[c]
                start = int(start)
                end = int(end)
                assert(start < len(token_start_indices))
                while previous_end < start:
                    subtoken_start = token_start_indices[previous_end]
                    subtoken_end = token_start_indices[previous_end + 1] \
                        if previous_end + 1 < len(token_start_indices) else len(tokens)
                    semiotic_classes.append((plain_cid, subtoken_start, subtoken_end))
                    previous_end += 1
                subtoken_start = token_start_indices[start]
                subtoken_end = token_start_indices[end] if end < len(token_start_indices) else len(tokens)
                semiotic_classes.append((cid, subtoken_start, subtoken_end))
                previous_end = end
            while previous_end < len(token_start_indices):
                subtoken_start = token_start_indices[previous_end]
                subtoken_end = token_start_indices[previous_end + 1] \
                    if previous_end + 1 < len(token_start_indices) else len(tokens)
                semiotic_classes.append((plain_cid, subtoken_start, subtoken_end))
                previous_end += 1
        if len(input_ids) > self._max_seq_length or len(semiotic_classes) > self._max_semiotic_length:
            return None
        example = BertExample(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            labels=labels,
            labels_mask=labels_mask,
            token_start_indices=token_start_indices,
            semiotic_classes=semiotic_classes,
            task=task,
            default_label=self._keep_tag_id,
        )
        example.pad_to_max_length(self._max_seq_length, self._max_semiotic_length, self._pad_id)
        return example

    def _split_to_wordpieces(self, tokens: List[str], labels: List[int]) -> Tuple[List[str], List[int], List[int]]:
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

    def _truncate_list(self, x: Union[List[str], List[int]]) -> Union[List[str], List[int]]:
        """Returns truncated version of x according to the self._max_seq_length."""
        # Save two slots for the first [CLS] token and the last [SEP] token.
        return x[: self._max_seq_length - 2]

    def _get_pad_id(self) -> int:
        """Returns the ID of the [PAD] token (or 0 if it's not in the vocab)."""
        try:
            return self._tokenizer.pad_token_id
        except KeyError:
            return 0


def read_input_file(
    example_builder: 'BertExampleBuilder',
    input_filename: str,
    infer: bool = False
) -> List['BertExample']:
    """Reads in Tab Separated Value file and converts to training/inference-ready examples.

    Args:
        example_builder: Instance of BertExampleBuilder
        input_filename: Path to the TSV input file.
        infer: Whether test files or not.

    Returns:
        examples: List of converted examples(features and Editing Tasks)
    """

    examples = []
    for i, (source, target, semiotic_info) in enumerate(yield_sources_and_targets(input_filename)):
        if len(examples) % 1000 == 0:
            logging.info("{} examples processed.".format(len(examples)))
        example = example_builder.build_bert_example(
            source, target, semiotic_info, infer
        )
        if example is None:
            continue
        examples.append(example)
    logging.info(f'Done. {len(examples)} examples converted.')
    return examples
