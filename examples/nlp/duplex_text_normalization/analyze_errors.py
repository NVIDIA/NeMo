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

"""
This script can be used to visualize the errors made by a (duplex) TN system.
More specifically, after running the evaluation script `duplex_text_normalization_test.py`,
a log file containing info about the errors will be generated. The location of this file
is determined by the argument `inference.errors_log_fp`. After that, we can use this
script to generate a HTML visualization.

USAGE Example:
# python analyze_errors.py                                    \
        --errors_log_fp=PATH_TO_ERRORS_LOG_FILE_PATH          \
        --visualization_fp=PATH_TO_VISUALIZATION_FILE_PATH
"""

from argparse import ArgumentParser
from typing import List

from nemo.collections.nlp.data.text_normalization import constants


# Longest Common Subsequence
def lcs(X, Y):
    """ Function for finding the longest common subsequence between two lists.
    In this script, this function is particular used for aligning between the
    ground-truth output string and the predicted string (for visualization purpose).
    Args:
        X: a list
        Y: a list

    Returns: a list which is the longest common subsequence between X and Y
    """
    m, n = len(X), len(Y)
    L = [[0 for x in range(n + 1)] for x in range(m + 1)]

    # Following steps build L[m+1][n+1] in bottom up fashion. Note
    # that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # Following code is used to print LCS
    index = L[m][n]

    # Create a character array to store the lcs string
    lcs = [''] * (index + 1)
    lcs[index] = ''

    # Start from the right-most-bottom-most corner and
    # one by one store characters in lcs[]
    i = m
    j = n
    while i > 0 and j > 0:

        # If current character in X[] and Y are same, then
        # current character is part of LCS
        if X[i - 1] == Y[j - 1]:
            lcs[index - 1] = X[i - 1]
            i -= 1
            j -= 1
            index -= 1

        # If not same, then find the larger of two and
        # go in the direction of larger value
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return lcs[:-1]


# Classes
class ErrorCase:
    """
    This class represents an error case

    Args:
        _input: Original input string
        target: Ground-truth target string
        pred: Predicted string
        mode: A string indicates the mode (i.e., constants.ITN_MODE or constants.TN_MODE)
    """

    def __init__(self, _input: str, target: str, pred: str, classes: str, mode: str):
        self._input = _input
        self.target = target
        self.pred = pred
        self.mode = mode
        self.classes = classes

        # Tokens
        self.target_tokens = self.target.split(' ')
        self.pred_tokens = self.pred.split(' ')

        # LCS
        lcs_tokens = lcs(self.target_tokens, self.pred_tokens)
        target_tokens_hightlight = [False] * len(self.target_tokens)
        pred_tokens_hightlight = [False] * len(self.pred_tokens)
        target_idx, pred_idx = 0, 0
        for token in lcs_tokens:
            while self.target_tokens[target_idx] != token:
                target_idx += 1
            while self.pred_tokens[pred_idx] != token:
                pred_idx += 1
            target_tokens_hightlight[target_idx] = True
            pred_tokens_hightlight[pred_idx] = True
            target_idx += 1
            pred_idx += 1

        # Spans
        self.target_spans = self.get_spans(target_tokens_hightlight)
        self.pred_spans = self.get_spans(pred_tokens_hightlight)

        # Determine unhighlighted target spans
        unhighlighted_target_spans = []
        for ix, t in enumerate(self.target_spans):
            if not t[-1]:
                unhighlighted_target_spans.append((ix, t))
        # Determine unhighlighted pred spans
        unhighlighted_pred_spans = []
        for ix, t in enumerate(self.pred_spans):
            if not t[-1]:
                unhighlighted_pred_spans.append((ix, t))

    @classmethod
    def from_lines(cls, lines: List[str], mode: str):
        """
        This method returns an instance of ErrorCase from raw string lines.

        Args:
            lines: A list of raw string lines for the error case.
            mode: A string indicates the mode (i.e., constants.ITN_MODE or constants.TN_MODE)

        Returns: an instance of ErrorCase.
        """
        for line in lines:
            if line.startswith('Original Input'):
                _input = line[line.find(':') + 1 :].strip()
            elif line.startswith('Predicted Str'):
                pred = line[line.find(':') + 1 :].strip()
            elif line.startswith('Ground-Truth'):
                target = line[line.find(':') + 1 :].strip()
            elif line.startswith('Ground Classes'):
                classes = line[line.find(':') + 1 :].strip()
        return cls(_input, target, pred, classes, mode)

    def get_html(self):
        """
        This method returns a HTML string representing this error case instance.
        Returns: a string contains the HTML representing this error case instance.
        """
        html_str = ''
        # Input
        input_form = 'Written' if self.mode == constants.TN_MODE else 'Spoken'
        padding_multiplier = 1 if self.mode == constants.TN_MODE else 2
        padding_spaces = ''.join(['&nbsp;'] * padding_multiplier)
        input_str = f'<b>[Input ({input_form})]{padding_spaces}</b>: {self._input}</br>\n'
        html_str += input_str + ' '
        # Target
        target_html = self.get_spans_html(self.target_spans, self.target_tokens)
        target_form = 'Spoken' if self.mode == constants.TN_MODE else 'Written'
        target_str = f'<b>[Target ({target_form})]</b>: {target_html}</br>\n'
        html_str += target_str + ' '
        # Pred
        pred_html = self.get_spans_html(self.pred_spans, self.pred_tokens)
        padding_multiplier = 10 if self.mode == constants.TN_MODE else 11
        padding_spaces = ''.join(['&nbsp;'] * padding_multiplier)
        pred_str = f'<b>[Prediction]{padding_spaces}</b>: {pred_html}</br>\n'
        html_str += pred_str + ' '
        # Classes
        padding_multiplier = 15 if self.mode == constants.TN_MODE else 16
        padding_spaces = ''.join(['&nbsp;'] * padding_multiplier)
        class_str = f'<b>[Classes]{padding_spaces}</b>: {self.classes}</br>\n'
        html_str += class_str + ' '
        # Space
        html_str += '</br>\n'
        return html_str

    def get_spans(self, tokens_hightlight):
        """
        This method extracts the list of spans.

        Args:
            tokens_hightlight: A list of boolean values where each value indicates whether a token needs to be hightlighted.

        Returns:
            spans: A list of spans. Each span is represented by a tuple of 3 elements: (1) Start Index (2) End Index (3) A boolean value indicating whether the span needs to be hightlighted.
        """
        spans, nb_tokens = [], len(tokens_hightlight)
        cur_start_idx, cur_bool_val = 0, tokens_hightlight[0]
        for idx in range(nb_tokens):
            if idx == nb_tokens - 1:
                if tokens_hightlight[idx] != cur_bool_val:
                    spans.append((cur_start_idx, nb_tokens - 2, cur_bool_val))
                    spans.append((nb_tokens - 1, nb_tokens - 1, tokens_hightlight[idx]))
                else:
                    spans.append((cur_start_idx, nb_tokens - 1, cur_bool_val))
            else:
                if tokens_hightlight[idx] != cur_bool_val:
                    spans.append((cur_start_idx, idx - 1, cur_bool_val))
                    cur_start_idx, cur_bool_val = idx, tokens_hightlight[idx]
        return spans

    def get_spans_html(self, spans, tokens):
        """
        This method generates a HTML string for a string sequence from its spans.

        Args:
            spans: A list of contiguous spans in a sequence. Each span is represented by a tuple of 3 elements: (1) Start Index (2) End Index (3) A boolean value indicating whether the span needs to be hightlighted.
            tokens: All tokens in the sequence
        Returns:
            html_str: A HTML string for the string sequence.
        """
        html_str = ''
        for start, end, type in spans:
            color = 'red' if type else 'black'
            span_tokens = tokens[start : end + 1]
            span_str = '<span style="color:{}">{}</span> '.format(color, ' '.join(span_tokens))
            html_str += span_str
        return html_str


# Main function for analysis
def analyze(errors_log_fp: str, visualization_fp: str):
    """
    This method generates a HTML visualization of the error cases logged in a log file.

    Args:
        errors_log_fp: Path to the error log file
        visualization_fp: Path to the output visualization file

    """
    # Read lines from errors log
    with open(errors_log_fp, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Process lines
    tn_error_cases, itn_error_cases = [], []
    for ix in range(0, len(lines), 8):
        mode_line = lines[ix]
        info_lines = lines[ix + 1 : ix + 7]
        # Append new error case
        if mode_line.startswith('Forward Problem'):
            mode = constants.TN_MODE
            tn_error_cases.append(ErrorCase.from_lines(info_lines, mode))
        elif mode_line.startswith('Backward Problem'):
            mode = constants.ITN_MODE
            itn_error_cases.append(ErrorCase.from_lines(info_lines, mode))

    # Basic stats
    print('---- Text Normalization ----')
    print('Number of TN errors: {}'.format(len(tn_error_cases)))

    print('---- Inverse Text Normalization ---- ')
    print('Number of ITN errors: {}'.format(len(itn_error_cases)))

    # Produce a visualization
    with open(visualization_fp, 'w+', encoding='utf-8') as f:
        # Appendix
        f.write('Appendix</br>')
        f.write('<a href="#tn_section">Text Normalization Analysis.</a></br>')
        f.write('<a href="#itn_section">Inverse Text Normalization Analysis.</a>')

        # TN Section
        f.write('<h2 id="tn_section">Text Normalization</h2>\n')
        for errorcase in tn_error_cases:
            f.write(errorcase.get_html())

        # ITN Section
        f.write('<h2 id="itn_section">Inverse Text Normalization</h2>\n')
        for errorcase in itn_error_cases:
            f.write(errorcase.get_html())


if __name__ == '__main__':
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--errors_log_fp', help='Path to the error log file', required=True)
    parser.add_argument('--visualization_fp', help='Path to the output visualization file', required=True)
    args = parser.parse_args()

    analyze(args.errors_log_fp, args.visualization_fp)
