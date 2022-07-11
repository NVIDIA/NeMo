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

import torch

from nemo.collections.nlp.parts.utils_funcs import tensor2list


class QAMetrics(object):
    @staticmethod
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    @staticmethod
    def white_space_fix(text):
        return " ".join(text.split())

    @staticmethod
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    @staticmethod
    def normalize_answer(s: str):
        """ Lower text and remove punctuation, articles and extra whitespace """

        return QAMetrics.white_space_fix(QAMetrics.remove_articles(QAMetrics.remove_punc(s.lower())))

    @staticmethod
    def _get_normalized_tokens(s: str):
        """ Get normalized tokens """
        if not s:
            return []
        return QAMetrics.normalize_answer(s).split()

    @staticmethod
    def get_one_f1(prediction: str, ground_truth: str):
        """ Computes f1 score between prediction and ground truth """

        prediction_tokens = QAMetrics._get_normalized_tokens(prediction)
        ground_truth_tokens = QAMetrics._get_normalized_tokens(ground_truth)
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())

        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        if len(ground_truth_tokens) == 0 or len(prediction_tokens) == 0:
            return int(ground_truth_tokens == prediction_tokens)
        if num_same == 0:
            return 0

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1

    @staticmethod
    def get_one_exact_match(prediction: str, ground_truth: str):
        """ Computes exact match between prediction and ground truth """

        return int(QAMetrics.normalize_answer(prediction) == QAMetrics.normalize_answer(ground_truth))

    @staticmethod
    def convert_dict_outputs_to_lists(outputs, keys):
        output_lists = [[] for _ in range(len(keys))]
        for output in outputs:
            for i, key in enumerate(keys):
                if type(output[key]) == torch.Tensor:
                    output_lists[i].extend(tensor2list(output[key]))
                else:
                    output_lists[i].extend(output[key])

        return output_lists
