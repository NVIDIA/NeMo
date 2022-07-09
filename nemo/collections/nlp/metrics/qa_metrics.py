# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import re
import string
from collections import Counter


class QAMetrics(object):
    @classmethod
    def normalize_answer(cls, s: str):
        """ Lower text and remove punctuation, articles and extra whitespace """

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def _get_tokens(cls, s: str):
        """ Get normalized tokens """
        if not s:
            return []
        return cls.normalize_answer(s).split()

    @classmethod
    def get_f1(cls, prediction: str, ground_truth: str):
        """ Computes f1 score between prediction and ground truth """

        prediction_tokens = cls._get_tokens(prediction)
        ground_truth_tokens = cls._get_tokens(ground_truth)
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())

        if len(ground_truth_tokens) == 0 or len(prediction_tokens) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(ground_truth_tokens == prediction_tokens)
        if num_same == 0:
            return 0

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1

    @classmethod
    def get_exact_match(cls, prediction: str, ground_truth: str):
        """ Computes exact match between prediction and ground truth """

        return int(cls.normalize_answer(prediction) == cls.normalize_answer(ground_truth))
