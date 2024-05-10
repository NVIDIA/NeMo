# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from torchmetrics import Accuracy, AveragePrecision, F1Score, MatthewsCorrCoef, PearsonCorrCoef, SpearmanCorrCoef
from torchmetrics.text.bert import BERTScore
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.edit import EditDistance
from torchmetrics.text.perplexity import Perplexity
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.sacre_bleu import SacreBLEUScore

from nemo.collections.common.metrics.classification_accuracy import ExactStringMatchMetric, TokenF1Score

__all__ = ['MetricStringToTorchMetric']

# Dictionary that maps a metric string name to its corresponding torchmetric class.

MetricStringToTorchMetric = {
    'accuracy': Accuracy,
    'average_precision': AveragePrecision,
    'f1': F1Score,
    'token_f1': TokenF1Score,
    'pearson_corr_coef': PearsonCorrCoef,
    'spearman_corr_coef': SpearmanCorrCoef,
    'matthews_corr_coef': MatthewsCorrCoef,
    'exact_string_match': ExactStringMatchMetric,
    'rouge': ROUGEScore,
    'bleu': BLEUScore,
    'sacre_bleu': SacreBLEUScore,
    'bertscore': BERTScore,
    'edit_distance': EditDistance,
    'perplexity': Perplexity,
}
