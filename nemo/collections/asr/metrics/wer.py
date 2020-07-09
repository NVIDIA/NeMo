# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List

import editdistance
import torch
from pytorch_lightning.metrics import TensorMetric

__all__ = ['word_error_rate', 'AverageTextWER']


def word_error_rate(hypotheses: List[str], references: List[str], use_cer=False) -> float:
    """
    Computes Average Word Error rate between two texts represented as
    corresponding lists of string. Hypotheses and references must have same
    length.
    Args:
      hypotheses: list of hypotheses
      references: list of references
      use_cer: bool, set True to enable cer
    Returns:
      (float) average word error rate
    """
    scores = 0
    words = 0
    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )
    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()
        words += len(r_list)
        scores += editdistance.eval(h_list, r_list)
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float('inf')
    return wer


class AverageTextWER(TensorMetric):
    """
    This metric computes nominator and denominator for Average Text Word Error Rate.
    When doing distributed training/evaluation the result of res=AverageTextWER(predictions, targets, target_lengths) calls
    will be all-reduced between all workers using SUM operations.
    Here contains two numbers res=[wer_nominator, wer_denominator]

    If used with PytorchLightning LightningModule, include wer_nominator and wer_denominator inside validation_step results.
    Then aggregate (sum) then at the end of validation epoch to correctly compute validation WER.

    Example:
        def validation_step(self, batch, batch_idx):
            ...
            wer_num, wer_denum = self.__wer(predictions, transcript, transcript_len)
            return {'val_loss': loss_value, 'val_wer_num': wer_num, 'val_wer_denum': wer_denum}

        def validation_epoch_end(self, outputs):
            ...
            wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
            wer_denum = torch.stack([x['val_wer_denum'] for x in outputs]).sum()
            tensorboard_logs = {'validation_loss': val_loss_mean, 'validation_avg_wer': wer_num / wer_denum}
            return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    Args:
        vocabulary:
        batch_dim_index:
        use_cer:
        ctc_decode:

    Returns:
        res: a torch.Tensor object with two elements: [wer_nominator, wer_denominator]. To correctly compute average
        text word error rate, compute wer=wer_nominator/wer_denominator
    """

    def __init__(self, vocabulary, batch_dim_index=0, use_cer=False, ctc_decode=True):
        super(AverageTextWER, self).__init__(name="AverageTextWER")
        self.batch_dim_index = batch_dim_index
        self.blank_id = len(vocabulary)
        self.labels_map = dict([(i, vocabulary[i]) for i in range(len(vocabulary))])
        self.use_cer = use_cer
        self.ctc_decode = ctc_decode

    def __ctc_decoder_predictions_tensor(self, predictions: torch.Tensor) -> List[str]:
        """
        Decodes a sequence of labels to words
        """
        hypotheses = []
        # Drop predictions to CPU
        prediction_cpu_tensor = predictions.long().cpu()
        # iterate over batch
        for ind in range(prediction_cpu_tensor.shape[self.batch_dim_index]):
            prediction = prediction_cpu_tensor[ind].detach().numpy().tolist()
            # CTC decoding procedure
            decoded_prediction = []
            previous = self.blank_id
            for p in prediction:
                if (p != previous or previous == self.blank_id) and p != self.blank_id:
                    decoded_prediction.append(p)
                previous = p
            hypothesis = ''.join([self.labels_map[c] for c in decoded_prediction])
            hypotheses.append(hypothesis)
        return hypotheses

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        words = 0.0
        scores = 0.0
        references = []
        with torch.no_grad():
            # prediction_cpu_tensor = tensors[0].long().cpu()
            targets_cpu_tensor = targets.long().cpu()
            tgt_lenths_cpu_tensor = target_lengths.long().cpu()

            # iterate over batch
            for ind in range(targets_cpu_tensor.shape[self.batch_dim_index]):
                tgt_len = tgt_lenths_cpu_tensor[ind].item()
                target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
                reference = ''.join([self.labels_map[c] for c in target])
                references.append(reference)
            if self.ctc_decode:
                hypotheses = self.__ctc_decoder_predictions_tensor(predictions)
            else:
                raise NotImplementedError("Implement me if you need non-CTC decode on predictions")

        for h, r in zip(hypotheses, references):
            if self.use_cer:
                h_list = list(h)
                r_list = list(r)
            else:
                h_list = h.split()
                r_list = r.split()
            words += len(r_list)
            # Compute Levenstein's distance
            scores += editdistance.eval(h_list, r_list)
        return torch.tensor([scores, words]).to(predictions.device)
