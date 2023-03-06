# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import numpy as np
import torch
from rouge_score import rouge_scorer
from sacrebleu import BLEU


class AccuracyScore(object):
    def get_score(self, ground_truth, predicted_text):
        corrects = 0
        for (pred, label) in zip(predicted_text, ground_truth):
            if pred == label:
                corrects += 1

        val_acc = corrects / len(ground_truth)
        val_acc = torch.tensor(val_acc).cuda()
        return {'accuracy': torch.tensor(val_acc).cuda()}


class BLEUScore(object):
    def __init__(self):
        self.scorer = BLEU()

    def get_score(self, ground_truth, predicted_text):
        return {
            'bleu_score': torch.tensor(
                self.scorer.corpus_score(predicted_text, [[i] for i in ground_truth],).score
            ).cuda()
        }


class ROUGEScores(object):
    def __init__(self, use_stemmer=True):
        self.rscorers = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=use_stemmer)

    def get_score(self, ground_truth, predicted_text):
        all_rouges = []
        for i in range(len(ground_truth)):
            scores = self.rscorers.score(predicted_text[i], ground_truth[i])
            all_rouges.append(
                [
                    scores['rouge1'].fmeasure,
                    scores['rouge2'].fmeasure,
                    scores['rouge3'].fmeasure,
                    scores['rougeL'].fmeasure,
                ]
            )

        all_rouges = np.mean(np.array(all_rouges), axis=0).tolist()

        return {
            'rouge_1_score': torch.tensor(all_rouges[0]).cuda(),
            'rouge_2_score': torch.tensor(all_rouges[1]).cuda(),
            'rouge_3_score': torch.tensor(all_rouges[2]).cuda(),
            'rouge_L_score': torch.tensor(all_rouges[3]).cuda(),
        }
