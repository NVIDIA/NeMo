# -*- coding: utf-8 -*-
# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
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

"""Build BERT Examples from text (source, target) pairs."""

import collections

from official_lasertagger import tagging

from nemo.collections.nlp.data.tokenizers import bert_tokenizer


class BertExample(object):
    """Class for training and inference examples for BERT.

    Attributes:
        editing_task: The EditingTask from which this example was created. Needed
            when realizing labels predicted for this example.
        features: Feature dictionary.
    """

    def __init__(
        self,
        input_ids,
        input_mask,
        segment_ids,
        tgt_ids,
        labels,
        labels_mask,
        token_start_indices,
        task,
        default_label,
    ):
        """Inputs to the example wrapper

        Args:
            input_ids: indices of tokens which constitute batches of masked text segments
            input_mask: bool tensor with 0s in place of source tokens to be masked
            segment_ids: bool tensor with 0's and 1's to denote the text segment type
            tgt_ids: indices of target tokens which constitute batches of masked text segments
            labels_mask: bool tensor with 0s in place of label tokens to be masked
            labels: indices of tokens which should be predicted from each of the
                corresponding target tokens in tgt_ids
            token_start_indices: the indices of the WordPieces that start a token.
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

        self.features = collections.OrderedDict(
            [
                ('input_ids', input_ids),
                ('input_mask', input_mask),
                ('segment_ids', segment_ids),
                ('tgt_ids', tgt_ids),
                ('labels', labels),
                ('labels_mask', labels_mask),
            ]
        )
        self._token_start_indices = token_start_indices
        self.editing_task = task
        self._default_label = default_label

    def pad_to_max_length(self, max_seq_length, pad_token_id):
        """Pad the feature vectors so that they all have max_seq_length.

        Args:
            max_seq_length: The length that features will have after padding.
            pad_token_id: input_ids feature is padded with this ID, other features
                with ID 0.
        """
        pad_len = max_seq_length - len(self.features['input_ids'])
        for key in self.features:
            if key == 'tgt_ids':
                self.features[key].extend([0] * (max_seq_length - len(self.features[key])))
            else:
                pad_id = pad_token_id if (key == 'input_ids') else 0
                self.features[key].extend([pad_id] * pad_len)
                if len(self.features[key]) != max_seq_length:
                    raise ValueError(
                        '{} has length {} (should be {}).'.format(key, len(self.features[key]), max_seq_length)
                    )

    def get_token_labels(self):
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

    def __init__(self, label_map, pretrained_model_name, max_seq_length, do_lower_case, converter):
        """Initializes an instance of BertExampleBuilder.

        Args:
            label_map: Mapping from tags to tag IDs.
            pretrained_model_name: Name of the pre-trained model.
            max_seq_length: Maximum sequence length.
            do_lower_case: Whether to lower case the input text. Should be True for
                uncased models and False for cased models.
            converter: Converter from text targets to tags.
        """
        self._label_map = label_map
        self._tokenizer = bert_tokenizer.NemoBertTokenizer(pretrained_model=pretrained_model_name)
        self._max_seq_length = max_seq_length
        self._converter = converter
        self._pad_id = self._get_pad_id()
        self._keep_tag_id = self._label_map['KEEP']
        self._task_tokens = collections.OrderedDict()

    def build_bert_example(
        self,
        sources,
        target=None,
        use_arbitrary_target_ids_for_infeasible_examples=False,
        save_tokens=True,
        infer=False,
    ):
        """Constructs a BERT Example.

        Args:
            sources: List of source texts.
            target: Target text or None when building an example during inference.
            use_arbitrary_target_ids_for_infeasible_examples: Whether to build an
                example with arbitrary target ids even if the target can't be obtained
                via tagging.

        Returns:
            BertExample, or None if the conversion from text to tags was infeasible
            and use_arbitrary_target_ids_for_infeasible_examples == False.
        """
        # Compute target labels.
        task = tagging.EditingTask(sources)
        if (target is not None) and (not infer):
            tags = self._converter.compute_tags(task, target)
            if not tags:
                if use_arbitrary_target_ids_for_infeasible_examples:
                    # Create a tag sequence [KEEP, DELETE, KEEP, DELETE, ...] which is
                    # unlikely to be predicted by chance.
                    tags = [
                        tagging.Tag('KEEP') if i % 2 == 0 else tagging.Tag('DELETE')
                        for i, _ in enumerate(task.source_tokens)
                    ]
                else:
                    return None
        else:
            # If target is not provided, we set all target labels to KEEP.
            tags = [tagging.Tag('KEEP') for _ in task.source_tokens]
        labels = [self._label_map[str(tag)] for tag in tags]
        tokens, labels, token_start_indices = self._split_to_wordpieces(task.source_tokens, labels)

        tokens = self._truncate_list(tokens)
        labels = self._truncate_list(labels)

        input_tokens = ['[CLS]'] + tokens + ['[SEP]']
        labels_mask = [0] + [1] * len(labels) + [0]
        labels = [0] + labels + [0]

        input_ids = self._tokenizer.tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        tgt_ids = self._truncate_list(self._tokenizer.text_to_ids(target))
        tgt_ids = [self._tokenizer.bos_id] + tgt_ids + [self._tokenizer.eos_id]

        if save_tokens:
            for i, t in enumerate(task.source_tokens):
                # Check of out of vocabulary tokens and save them
                if self._tokenizer.token_to_id(t) == 100:
                    self._task_tokens[t] = None

        example = BertExample(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            tgt_ids=tgt_ids,
            labels=labels,
            labels_mask=labels_mask,
            token_start_indices=token_start_indices,
            task=task,
            default_label=self._keep_tag_id,
        )
        example.pad_to_max_length(self._max_seq_length, self._pad_id)
        return example

    def get_special_tokens_and_ids(self):
        '''Returns list of additional out-of-vocab special tokens in test files
        used later by Editing Task during inference.
        '''
        return list(self._task_tokens.keys())

    def _split_to_wordpieces(self, tokens, labels):
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
            pieces = self._tokenizer.tokenizer.tokenize(token)
            bert_tokens.extend(pieces)
            bert_labels.extend([labels[i]] * len(pieces))
        return bert_tokens, bert_labels, token_start_indices

    def _truncate_list(self, x):
        """Returns truncated version of x according to the self._max_seq_length."""
        # Save two slots for the first [CLS] token and the last [SEP] token.
        return x[: self._max_seq_length - 2]

    def _get_pad_id(self):
        """Returns the ID of the [PAD] token (or 0 if it's not in the vocab)."""
        try:
            return self._tokenizer.tokens_to_ids(['[PAD]'])[0]
        except KeyError:
            return 0
