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

import torch
from pytorch_lightning.metrics import TensorMetric

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


def __ctc_decoder_predictions_tensor(tensor, tokenizer: TokenizerSpec):
    """
    Decodes a sequence of labels to words
    """
    blank_id = tokenizer.tokenizer.vocab_size
    hypotheses = []
    prediction_cpu_tensor = tensor.long().cpu()
    # iterate over batch
    for ind in range(prediction_cpu_tensor.shape[0]):
        prediction = prediction_cpu_tensor[ind].detach().numpy().tolist()
        # CTC decoding procedure
        decoded_prediction = []
        previous = blank_id  # id of a blank symbol
        for p in prediction:
            if (p != previous or previous == blank_id) and p != blank_id:
                decoded_prediction.append(p)
            previous = p
        hypothesis = tokenizer.ids_to_text(decoded_prediction)
        hypotheses.append(hypothesis)
    return hypotheses


def monitor_asr_train_progress(tensors: list, tokenizer: TokenizerSpec, eval_metric='WER', ctc_decode=True):
    """
    Takes output of greedy ctc decoder and performs ctc decoding algorithm to
    remove duplicates and special symbol. Prints sample to screen, computes
    and logs AVG WER to console
    Args:
      tensors: A list of 3 tensors (predictions, targets, target_lengths)
      labels: A list of labels
      eval_metric: An optional string from 'WER', 'CER'. Defaults to 'WER'.
      ctc_decode: Bool whether CTC or RNNT decoding should be applied.
        Currently unimplemented.
    Returns:
      batch wer, hypothesis and reference from the first batch element
    """
    references = []

    with torch.no_grad():
        # prediction_cpu_tensor = tensors[0].long().cpu()
        targets_cpu_tensor = tensors[1].long().cpu()
        tgt_lenths_cpu_tensor = tensors[2].long().cpu()

        # iterate over batch
        for ind in range(targets_cpu_tensor.shape[0]):
            tgt_len = tgt_lenths_cpu_tensor[ind].item()
            target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
            reference = tokenizer.ids_to_text(target)
            references.append(reference)
        if ctc_decode:
            hypotheses = __ctc_decoder_predictions_tensor(tensors[0], tokenizer=tokenizer)
        else:
            raise NotImplementedError("Currently, we only support WER for CTC models' output")

    eval_metric = eval_metric.upper()
    if eval_metric not in {'WER', 'CER'}:
        raise ValueError('eval_metric must be \'WER\' or \'CER\'')
    use_cer = True if eval_metric == 'CER' else False
    wer = word_error_rate(hypotheses, references, use_cer=use_cer)
    return wer, hypotheses[0], references[0]


class WordErrorRateBPE(TensorMetric):
    def __init__(self, tokenizer: TokenizerSpec, ctc_decode=True):
        super(WordErrorRateBPE, self).__init__(name="WER-BPE")
        self.tokenizer = tokenizer
        self.ctc_decode = ctc_decode

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        wer, _, _ = monitor_asr_train_progress(
            tensors=[predictions, targets, target_lengths],
            tokenizer=self.tokenizer,
            eval_metric='WER',
            ctc_decode=self.ctc_decode,
        )
        wer = torch.tensor(wer, dtype=predictions.dtype, device=predictions.device)
        return wer
