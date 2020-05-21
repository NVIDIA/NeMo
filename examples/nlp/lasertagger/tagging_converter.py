# coding=utf-8
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

"""Conversion from training target text into target tags.

The conversion algorithm from (source, target) pairs to (source, target_tags)
pairs is described in Algorithm 1 of the LaserTagger paper
(https://arxiv.org/abs/1909.01187).
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tagging
import utils

from typing import Iterable, Mapping, Sequence, Set, Text, Tuple


class TaggingConverter(object):
  """Converter from training target texts into tagging format."""

  def __init__(self, phrase_vocabulary, do_swap=True):
    """Initializes an instance of TaggingConverter.

    Args:
      phrase_vocabulary: Iterable of phrase vocabulary items (strings).
      do_swap: Whether to enable the SWAP tag.
    """
    self._phrase_vocabulary = set(
        phrase.lower() for phrase in phrase_vocabulary)
    self._do_swap = do_swap
    # Maximum number of tokens in an added phrase (inferred from the
    # vocabulary).
    self._max_added_phrase_length = 0
    # Set of tokens that are part of a phrase in self.phrase_vocabulary.
    self._token_vocabulary = set()
    for phrase in self._phrase_vocabulary:
      tokens = utils.get_token_list(phrase)
      self._token_vocabulary |= set(tokens)
      if len(tokens) > self._max_added_phrase_length:
        self._max_added_phrase_length = len(tokens)

  def compute_tags(self, task,
                   target):
    """Computes tags needed for converting the source into the target.

    Args:
      task: tagging.EditingTask that specifies the input.
      target: Target text.

    Returns:
      List of tagging.Tag objects. If the source couldn't be converted into the
      target via tagging, returns an empty list.
    """
    target_tokens = utils.get_token_list(target.lower())
    tags = self._compute_tags_fixed_order(task.source_tokens, target_tokens)
    # If conversion fails, try to obtain the target after swapping the source
    # order.
    if not tags and len(task.sources) == 2 and self._do_swap:
      swapped_task = tagging.EditingTask(task.sources[::-1])
      tags = self._compute_tags_fixed_order(swapped_task.source_tokens,
                                            target_tokens)
      if tags:
        tags = (tags[swapped_task.first_tokens[1]:] +
                tags[:swapped_task.first_tokens[1]])
        # We assume that the last token (typically a period) is never deleted,
        # so we can overwrite the tag_type with SWAP (which keeps the token,
        # moving it and the sentence it's part of to the end).
        tags[task.first_tokens[1] - 1].tag_type = tagging.TagType.SWAP
    return tags

  def _compute_tags_fixed_order(self, source_tokens, target_tokens):
    """Computes tags when the order of sources is fixed.

    Args:
      source_tokens: List of source tokens.
      target_tokens: List of tokens to be obtained via edit operations.

    Returns:
      List of tagging.Tag objects. If the source couldn't be converted into the
      target via tagging, returns an empty list.
    """
    tags = [tagging.Tag('DELETE') for _ in source_tokens]
    # Indices of the tokens currently being processed.
    source_token_idx = 0
    target_token_idx = 0
    while target_token_idx < len(target_tokens):
      tags[source_token_idx], target_token_idx = self._compute_single_tag(
          source_tokens[source_token_idx], target_token_idx, target_tokens)
      # If we're adding a phrase and the previous source token(s) were deleted,
      # we could add the phrase before a previously deleted token and still get
      # the same realized output. For example:
      #    [DELETE, DELETE, KEEP|"what is"]
      # and
      #    [DELETE|"what is", DELETE, KEEP]
      # Would yield the same realized output. Experimentally, we noticed that
      # the model works better / the learning task becomes easier when phrases
      # are always added before the first deleted token. Also note that in the
      # current implementation, this way of moving the added phrase backward is
      # the only way a DELETE tag can have an added phrase, so sequences like
      # [DELETE|"What", DELETE|"is"] will never be created.
      if tags[source_token_idx].added_phrase:
        first_deletion_idx = self._find_first_deletion_idx(
            source_token_idx, tags)
        if first_deletion_idx != source_token_idx:
          tags[first_deletion_idx].added_phrase = (
              tags[source_token_idx].added_phrase)
          tags[source_token_idx].added_phrase = ''
      source_token_idx += 1
      if source_token_idx >= len(tags):
        break

    # If all target tokens have been consumed, we have found a conversion and
    # can return the tags. Note that if there are remaining source tokens, they
    # are already marked deleted when initializing the tag list.
    if target_token_idx >= len(target_tokens):
      return tags
    return []

  def _compute_single_tag(
      self, source_token, target_token_idx,
      target_tokens):
    """Computes a single tag.

    The tag may match multiple target tokens (via tag.added_phrase) so we return
    the next unmatched target token.

    Args:
      source_token: The token to be tagged.
      target_token_idx: Index of the current target tag.
      target_tokens: List of all target tokens.

    Returns:
      A tuple with (1) the computed tag and (2) the next target_token_idx.
    """
    source_token = source_token.lower()
    target_token = target_tokens[target_token_idx].lower()
    if source_token == target_token:
      return tagging.Tag('KEEP'), target_token_idx + 1

    added_phrase = ''
    for num_added_tokens in range(1, self._max_added_phrase_length + 1):
      if target_token not in self._token_vocabulary:
        break
      added_phrase += (' ' if added_phrase else '') + target_token
      next_target_token_idx = target_token_idx + num_added_tokens
      if next_target_token_idx >= len(target_tokens):
        break
      target_token = target_tokens[next_target_token_idx].lower()
      if (source_token == target_token and
          added_phrase in self._phrase_vocabulary):
        return tagging.Tag('KEEP|' + added_phrase), next_target_token_idx + 1
    return tagging.Tag('DELETE'), target_token_idx

  def _find_first_deletion_idx(self, source_token_idx, tags):
    """Finds the start index of a span of deleted tokens.

    If `source_token_idx` is preceded by a span of deleted tokens, finds the
    start index of the span. Otherwise, returns `source_token_idx`.

    Args:
      source_token_idx: Index of the current source token.
      tags: List of tags.

    Returns:
      The index of the first deleted token preceding `source_token_idx` or
      `source_token_idx` if there are no deleted tokens right before it.
    """
    # Backtrack until the beginning of the tag sequence.
    for idx in range(source_token_idx, 0, -1):
      if tags[idx - 1].tag_type != tagging.TagType.DELETE:
        return idx
    return 0


def get_phrase_vocabulary_from_label_map(
    label_map):
  """Extract the set of all phrases from label map.

  Args:
    label_map: Mapping from tags to tag IDs.

  Returns:
    Set of all phrases appearing in the label map.
  """
  phrase_vocabulary = set()
  for label in label_map.keys():
    tag = tagging.Tag(label)
    if tag.added_phrase:
      phrase_vocabulary.add(tag.added_phrase)
  return phrase_vocabulary
