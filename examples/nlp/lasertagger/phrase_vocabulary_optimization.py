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

# Lint as: python3

"""
This file contains code artifacts adapted from the original implementation:
https://github.com/google-research/lasertagger/blob/master/phrase_vocabulary_optimization.py
"""

"""Optimizes the vocabulary of phrases that can be added by LaserTagger.

The goal is to find a fixed-size set of phrases that cover as many training
examples as possible. Based on the phrases, saves a file containing all possible
tags to be predicted and another file reporting the percentage of covered
training examples with different vocabulary sizes.
"""

import collections
from typing import Sequence, Text

import numpy as np
import scipy.sparse
from official_lasertagger import utils

from nemo import logging
from nemo.utils import NemoArgParser


def parse_args():
    '''
    LaserTagger Phrase Vocabulary Optimization argument parser
    '''
    parser = NemoArgParser(description="LaserTagger Phrase Vocabulary Optimization")
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the input file containing source-target pairs from which the vocabulary is optimized.",
    )
    parser.add_argument(
        "--max_input_examples",
        type=int,
        help="At most this many examples from the 'input_file' are used for optimizing the vocabulary.",
        default=50000,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to the resulting file with all possible tags. Coverage numbers will be written to a separate file which has the same path but '.log' appended to it.",
    )
    parser.add_argument("--enable_swap_tag", type=bool, help="Whether to enable the SWAP tag.", default=True)
    parser.add_argument(
        '--vocabulary_size', type=int, help='Number of phrases to include in the vocabulary.', default=500,
    )
    parser.add_argument(
        "--num_extra_statistics",
        type=int,
        help="Number of extra phrases that are not included in the vocabulary but for which we compute the coverage numbers. These numbers help determining whether the vocabulary size should have been larger.",
        default=100,
    )

    return parser.parse_args()


def _compute_lcs(source, target):
    """Computes the Longest Common Subsequence (LCS).

  Description of the dynamic programming algorithm:
  https://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
	source: List of source tokens.
	target: List of target tokens.

  Returns:
	List of tokens in the LCS.
  """
    table = _lcs_table(source, target)
    return _backtrack(table, source, target, len(source), len(target))


def _lcs_table(source, target):
    """Returns the Longest Common Subsequence dynamic programming table."""
    rows = len(source)
    cols = len(target)
    lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if source[i - 1] == target[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
    return lcs_table


def _backtrack(table, source, target, i, j):
    """Backtracks the Longest Common Subsequence table to reconstruct the LCS.

  Args:
	table: Precomputed LCS table.
	source: List of source tokens.
	target: List of target tokens.
	i: Current row index.
	j: Current column index.

  Returns:
	List of tokens corresponding to LCS.
  """
    if i == 0 or j == 0:
        return []
    if source[i - 1] == target[j - 1]:
        # Append the aligned token to output.
        return _backtrack(table, source, target, i - 1, j - 1) + [target[j - 1]]
    if table[i][j - 1] > table[i - 1][j]:
        return _backtrack(table, source, target, i, j - 1)
    else:
        return _backtrack(table, source, target, i - 1, j)


def _get_added_phrases(source: Text, target: Text) -> Sequence[Text]:
    """Computes the phrases that need to be added to the source to get the target.

  This is done by aligning each token in the LCS to the first match in the
  target and checking which phrases in the target remain unaligned.

  TODO(b/142853960): The LCS tokens should ideally be aligned to consecutive
  target tokens whenever possible, instead of aligning them always to the first
  match. This should result in a more meaningful phrase vocabulary with a higher
  coverage.

  Note that the algorithm is case-insensitive and the resulting phrases are
  always lowercase.

  Args:
	source: Source text.
	target: Target text.

  Returns:
	List of added phrases.
  """
    source_tokens = utils.get_token_list(source.lower())
    target_tokens = utils.get_token_list(target.lower())
    kept_tokens = _compute_lcs(source_tokens, target_tokens)
    added_phrases = []
    # Index of the `kept_tokens` element that we are currently looking for.
    kept_idx = 0
    phrase = []
    for token in target_tokens:
        if kept_idx < len(kept_tokens) and token == kept_tokens[kept_idx]:
            kept_idx += 1
            if phrase:
                added_phrases.append(' '.join(phrase))
                phrase = []
        else:
            phrase.append(token)
    if phrase:
        added_phrases.append(' '.join(phrase))
    return added_phrases


def _added_token_counts(data_iterator, try_swapping, max_input_examples=10000):
    """Computes how many times different phrases have to be added.

  Args:
	data_iterator: Iterator to yield source lists and targets. See function
	  yield_sources_and_targets in utils.py for the available iterators. The
	  strings in the source list will be concatenated, possibly after swapping
	  their order if swapping is enabled.
	try_swapping: Whether to try if swapping sources results in less added text.
	max_input_examples: Maximum number of examples to be read from the iterator.

  Returns:
	Tuple (collections.Counter for phrases, added phrases for each example).
  """
    phrase_counter = collections.Counter()
    num_examples = 0
    all_added_phrases = []
    for sources, target in data_iterator:
        if num_examples >= max_input_examples:
            break
        if num_examples % 1000 == 0:
            logging.info("{} examples processed.".format(num_examples))
        added_phrases = _get_added_phrases(' '.join(sources), target)
        if try_swapping and len(sources) == 2:
            added_phrases_swap = _get_added_phrases(' '.join(sources[::-1]), target)
            # If we can align more and have to add less after swapping, we assume that
            # the sources would be swapped during conversion.
            if len(''.join(added_phrases_swap)) < len(''.join(added_phrases)):
                added_phrases = added_phrases_swap
        for phrase in added_phrases:
            phrase_counter[phrase] += 1
        all_added_phrases.append(added_phrases)
        num_examples += 1
    logging.info(f'{num_examples} examples processed.\n')
    return phrase_counter, all_added_phrases


def _construct_added_phrases_matrix(all_added_phrases, phrase_counter):
    """Constructs a sparse phrase occurrence matrix.

  Examples are on rows and phrases on columns.

  Args:
	all_added_phrases: List of lists of added phrases (one list per example).
	phrase_counter: Frequence of each unique added phrase.

  Returns:
	Sparse boolean matrix whose element (i, j) indicates whether example i
	contains the added phrase j. Columns start from the most frequent phrase.
  """
    phrase_2_idx = {tup[0]: i for i, tup in enumerate(phrase_counter.most_common())}
    matrix = scipy.sparse.dok_matrix((len(all_added_phrases), len(phrase_2_idx)), dtype=np.bool)
    for i, added_phrases in enumerate(all_added_phrases):
        for phrase in added_phrases:
            phrase_idx = phrase_2_idx[phrase]
            matrix[i, phrase_idx] = True
    # Convert to CSC format to support more efficient column slicing.
    return matrix.tocsc()


def _count_covered_examples(matrix, vocabulary_size):
    """Returns the number of examples whose added phrases are in the vocabulary.

  This assumes the vocabulary is created simply by selecting the
  `vocabulary_size` most frequent phrases.

  Args:
	matrix: Phrase occurrence matrix with the most frequent phrases on the
	  left-most columns.
	vocabulary_size: Number of most frequent phrases to include in the
	  vocabulary.
  """
    # Ignore the `vocabulary_size` most frequent (i.e. leftmost) phrases (i.e.
    # columns) and count the rows with zero added phrases.
    return (matrix[:, vocabulary_size:].sum(axis=1) == 0).sum()


if __name__ == "__main__":

    args = parse_args()

    data_iterator = utils.yield_sources_and_targets(args.input_file)
    phrase_counter, all_added_phrases = _added_token_counts(
        data_iterator, args.enable_swap_tag, args.max_input_examples
    )
    matrix = _construct_added_phrases_matrix(all_added_phrases, phrase_counter)
    num_examples = len(all_added_phrases)

    statistics_file = args.output_file + '.log'
    with open(args.output_file, 'w') as writer:
        with open(statistics_file, 'w') as stats_writer:
            stats_writer.write('Idx\tFrequency\tCoverage (%)\tPhrase\n')
            writer.write('KEEP\n')
            writer.write('DELETE\n')
            if args.enable_swap_tag:
                writer.write('SWAP\n')
            for i, (phrase, count) in enumerate(
                phrase_counter.most_common(args.vocabulary_size + args.num_extra_statistics)
            ):
                # Write tags.
                if i < args.vocabulary_size:
                    writer.write(f'KEEP|{phrase}\n')
                    writer.write(f'DELETE|{phrase}\n')
                # Write statistics.
                coverage = 100.0 * _count_covered_examples(matrix, i + 1) / num_examples
                stats_writer.write(f'{i+1}\t{count}\t{coverage:.2f}\t{phrase}\n')
    logging.info(f'Wrote tags to: {args.output_file}')
    logging.info(f'Wrote coverage numbers to: {statistics_file}')
