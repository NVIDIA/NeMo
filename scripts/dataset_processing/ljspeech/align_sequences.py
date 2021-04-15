# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

# Biopython is currently released under the "Biopython License Agreement" (given in full below).
# Unless stated otherwise in individual file headers, all Biopython's files are under the "Biopython License Agreement".
#
# Some files are explicitly dual licensed under your choice of the "Biopython License Agreement" or the
# "BSD 3-Clause License" (both given in full below).
# This is with the intention of later offering all of Biopython under this dual licensing approach.
#
# Biopython License Agreement
# Permission to use, copy, modify, and distribute this software and its documentation with or without modifications
# and for any purpose and without fee is hereby granted, provided that any copyright notices appear in all copies
# and that both those copyright notices and this permission notice appear in supporting documentation, and that the
# names of the contributors or copyright holders not be used in advertising or publicity pertaining to distribution
# of the software without specific prior permission.
#
# THE CONTRIBUTORS AND COPYRIGHT HOLDERS OF THIS SOFTWARE DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
# INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
# IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#
# BSD 3-Clause License
# Copyright (c) 1999-2020, The Biopython Contributors All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Adapted from https://github.com/biopython/biopython/blob/fce4b11b4b8e414f1bf093a76e04a3260d782905/Bio/pairwise2.py
"""
from collections import namedtuple


def align(seq_a, seq_B, match_fn, gap_char=['-'], one_alignment_only=True):
    """ Return optimal alignments between two sequences """
    score_matrix, trace_matrix, best_score = make_score_matrix_generic(
        seq_a, seq_B, match_fn)

    starts = find_start(score_matrix, best_score)

    # Recover the alignments and return them.
    alignments = recover_alignments(
        seq_a, seq_B, starts, best_score, score_matrix, trace_matrix,
        gap_char, one_alignment_only)

    return alignments


def make_score_matrix_generic(seq_a, seq_B, match_fn):
    """ Generate a score and traceback matrix. """
    # Create the score and traceback matrices. These should be in the shape:
    # seq_a (down) x seq_B (across)
    lenA, lenB = len(seq_a), len(seq_B)
    score_matrix, trace_matrix = [], []
    for i in range(lenA + 1):
        score_matrix.append([None] * (lenB + 1))
        trace_matrix.append([None] * (lenB + 1))

    # Initialize first row and column with gap scores. This is like opening up
    # i gaps at the beginning of sequence A or B.
    for i in range(lenA + 1):
        score_matrix[i][0] = 0
    for i in range(lenB + 1):
        score_matrix[0][i] = 0

    # Fill in the score matrix.  Each position in the matrix represents an
    # alignment between a character from sequence A to one in sequence B.  As I
    # iterate through the matrix, find the alignment by choosing the best of:
    #    1) extending a previous alignment without gaps
    #    2) adding a gap in seq_a
    #    3) adding a gap in seq_B
    for row in range(1, lenA + 1):
        for col in range(1, lenB + 1):
            # First, calculate the score that would occur by extending
            # the alignment without gaps.
            nogap_score = (score_matrix[row - 1][col - 1]
                           + match_fn(seq_a[row - 1], seq_B[col - 1]))

            # Try to find a better score by opening gaps in seq_a.
            # Do this by checking alignments from each column in the row.
            # Each column represents a different character to align from, and
            # thus a different length gap.
            # Although the gap function does not distinguish between opening and
            # extending a gap, we distinguish them for the backtrace.
            row_open = score_matrix[row][col - 1]
            row_extend = max(score_matrix[row][x] for x in range(col))

            # Try to find a better score by opening gaps in seq_B.
            col_open = score_matrix[row - 1][col]
            col_extend = max(score_matrix[x][col] for x in range(row))

            best_score = max(nogap_score, row_open, row_extend, col_open, col_extend)
            score_matrix[row][col] = best_score

            # The backtrace is encoded binary.
            trace_score = 0
            if rint(nogap_score) == rint(best_score):
                trace_score += 2
            if rint(row_open) == rint(best_score):
                trace_score += 1
            if rint(row_extend) == rint(best_score):
                trace_score += 8
            if rint(col_open) == rint(best_score):
                trace_score += 4
            if rint(col_extend) == rint(best_score):
                trace_score += 16
            trace_matrix[row][col] = trace_score

    return score_matrix, trace_matrix, best_score


def recover_alignments(seq_a, seq_B, starts, best_score, score_matrix,
                       trace_matrix, gap_char, one_alignment_only,
                       max_alignments=10):
    """ Backtrack and return a list of alignments. """
    ali_seqA, ali_seqB = seq_a[0:0], seq_B[0:0]
    tracebacks = []
    in_process = []

    for start in starts:
        score, (row, col) = start
        begin = 0
        end = None
        in_process += [
            (ali_seqA, ali_seqB, end, row, col, False, trace_matrix[row][col])
        ]
    while in_process and len(tracebacks) < max_alignments:
        # Although we allow a gap in seqB to be followed by a gap in seqA,
        # we don't want to allow it the other way round, since this would
        # give redundant alignments of type: A-  vs.  -A
        #                                    -B       B-
        # Thus we need to keep track if a gap in seqA was opened (col_gap)
        # and stop the backtrace (dead_end) if a gap in seqB follows.
        dead_end = False
        ali_seqA, ali_seqB, end, row, col, col_gap, trace = in_process.pop()

        while (row > 0 or col > 0) and not dead_end:
            cache = (ali_seqA[:], ali_seqB[:], end, row, col, col_gap)

            # If trace is empty we have reached at least one border of the
            # matrix or the end of a local alignment. Just add the rest of
            # the sequence(s) and fill with gaps if necessary.
            if not trace:
                if col and col_gap:
                    dead_end = True
                else:
                    ali_seqA, ali_seqB = finish_backtrace(
                        seq_a, seq_B, ali_seqA, ali_seqB, row, col,
                        gap_char)
                break
            elif trace % 2 == 1:  # = row open = open gap in seqA
                trace -= 1
                if col_gap:
                    dead_end = True
                else:
                    col -= 1
                    ali_seqA += gap_char
                    ali_seqB += seq_B[col:col + 1]
                    col_gap = False
            elif trace % 4 == 2:  # = match/mismatch of seqA with seqB
                trace -= 2
                row -= 1
                col -= 1
                ali_seqA += seq_a[row:row + 1]
                ali_seqB += seq_B[col:col + 1]
                col_gap = False
            elif trace % 8 == 4:  # = col open = open gap in seqB
                trace -= 4
                row -= 1
                ali_seqA += seq_a[row:row + 1]
                ali_seqB += gap_char
                col_gap = True
            elif trace in (8, 24):  # = row extend = extend gap in seqA
                trace -= 8
                if col_gap:
                    dead_end = True
                else:
                    col_gap = False
                    # We need to find the starting point of the extended gap
                    x = find_gap_open(
                        seq_a, seq_B, ali_seqA, ali_seqB, end, row,
                        col, col_gap, gap_char, score_matrix, trace_matrix,
                        in_process, col, row, "col", best_score)
                    ali_seqA, ali_seqB, row, col, in_process, dead_end = x
            elif trace == 16:  # = col extend = extend gap in seqB
                trace -= 16
                col_gap = True
                x = find_gap_open(
                    seq_a, seq_B, ali_seqA, ali_seqB, end, row, col,
                    col_gap, gap_char, score_matrix, trace_matrix, in_process,
                    row, col, "row", best_score)
                ali_seqA, ali_seqB, row, col, in_process, dead_end = x

            if trace:  # There is another path to follow...
                cache += (trace,)
                in_process.append(cache)
            trace = trace_matrix[row][col]
        if not dead_end:
            tracebacks.append(
                (ali_seqA[::-1], ali_seqB[::-1], score, begin, end))
            if one_alignment_only:
                break
    return clean_alignments(tracebacks)


def find_start(score_matrix, best_score):
    """Return a list of starting points (score, (row, col)) (PRIVATE).
    Indicating every possible place to start the tracebacks.
    """
    nrows, ncols = len(score_matrix), len(score_matrix[0])
    # In this implementation of the global algorithm, the start will always be
    # the bottom right corner of the matrix.
    starts = [(best_score, (nrows - 1, ncols - 1))]
    return starts


def clean_alignments(alignments):
    """
    Take a list of alignments and return a cleaned version.
    Remove duplicates, make sure begin and end are set correctly, remove empty
    alignments.
    """
    Alignment = namedtuple("Alignment", ("seqA, seqB, score, start, end"))
    unique_alignments = []
    for align in alignments:
        if align not in unique_alignments:
            unique_alignments.append(align)
    i = 0
    while i < len(unique_alignments):
        seqA, seqB, score, begin, end = unique_alignments[i]
        # Make sure end is set reasonably.
        if end is None:  # global alignment
            end = len(seqA)
        elif end < 0:
            end = end + len(seqA)
        # If there's no alignment here, get rid of it.
        if begin >= end:
            del unique_alignments[i]
            continue
        unique_alignments[i] = Alignment(seqA, seqB, score, begin, end)
        i += 1
    return unique_alignments


def finish_backtrace(seq_a, seq_B, ali_seqA, ali_seqB, row, col, gap_char):
    """ Add remaining sequences and fill with gaps if necessary. """
    if row:
        ali_seqA += seq_a[row - 1::-1]
    if col:
        ali_seqB += seq_B[col - 1::-1]
    if row > col:
        ali_seqB += gap_char * (len(ali_seqA) - len(ali_seqB))
    elif col > row:
        ali_seqA += gap_char * (len(ali_seqB) - len(ali_seqA))
    return ali_seqA, ali_seqB


def find_gap_open(seq_a, seq_B, ali_seqA, ali_seqB, end, row, col, col_gap,
                  gap_char, score_matrix, trace_matrix, in_process, target,
                  index, direction, best_score):
    """ Find the starting point(s) of the extended gap. """
    dead_end = False
    target_score = score_matrix[row][col]
    for n in range(target):
        if direction == "col":
            col -= 1
            ali_seqA += gap_char
            ali_seqB += seq_B[col: col + 1]
        else:
            row -= 1
            ali_seqA += seq_a[row: row + 1]
            ali_seqB += gap_char
        actual_score = score_matrix[row][col]
        if rint(actual_score) == rint(target_score) and n > 0:
            if not trace_matrix[row][col]:
                break
            else:
                in_process.append((
                    ali_seqA[:], ali_seqB[:], end, row, col, col_gap,
                    trace_matrix[row][col]))
        if not trace_matrix[row][col]:
            dead_end = True
    return ali_seqA, ali_seqB, row, col, in_process, dead_end


def rint(x, precision=1000):
    """Print number with declared precision."""
    return int(x * precision + 0.5)
