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

from typing import Literal, Optional, Sequence, Union

import torch
from torchmetrics.functional.text.bleu import _bleu_score_compute
from torchmetrics.text import SacreBLEUScore

from nemo.collections.asr.parts.submodules.ctc_decoding import AbstractCTCDecoding
from nemo.collections.asr.parts.submodules.multitask_decoding import AbstractMultiTaskDecoding
from nemo.collections.asr.parts.submodules.rnnt_decoding import AbstractRNNTDecoding
from nemo.utils import logging

__all__ = ['BLEU']


def move_dimension_to_the_front(tensor, dim_index):
    all_dims = list(range(tensor.ndim))
    return tensor.permute(*([dim_index] + all_dims[:dim_index] + all_dims[dim_index + 1 :]))


# TODO: Add documentation
class BLEU(SacreBLEUScore):

    full_state_update: bool = True

    def __init__(
        self,
        decoding: Union[AbstractCTCDecoding, AbstractRNNTDecoding, AbstractMultiTaskDecoding],
        tokenize: Literal["none", "13a", "zh", "intl", "char"] = "13a",
        n_gram: int = 4,
        lowercase: bool = False,
        weights: Optional[Sequence[float]] = None,
        smooth: bool = False,
        log_prediction=True,
        batch_dim_index=0,
        dist_sync_on_step=False,
    ):
        super().__init__(
            tokenize=tokenize,
            n_gram=n_gram,
            lowercase=lowercase,
            weights=weights,
            smooth=smooth,
            dist_sync_on_step=dist_sync_on_step,
        )
        self.has_spl_tokens = False
        self.decoding = decoding
        self.decode = None
        if isinstance(self.decoding, AbstractRNNTDecoding):
            self.decode = lambda predictions, predictions_lengths, predictions_mask, input_ids, targets: self.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=predictions, encoded_lengths=predictions_lengths
            )
        elif isinstance(self.decoding, AbstractCTCDecoding):
            self.decode = lambda predictions, predictions_lengths, predictions_mask, input_ids, targets: self.decoding.ctc_decoder_predictions_tensor(
                decoder_outputs=predictions,
                decoder_lengths=predictions_lengths,
                fold_consecutive=self.fold_consecutive,
            )
        elif isinstance(self.decoding, AbstractMultiTaskDecoding):
            self.has_spl_tokens = True
            self.decode = lambda predictions, prediction_lengths, predictions_mask, input_ids, targets: self.decoding.decode_predictions_tensor(
                encoder_hidden_states=predictions,
                encoder_input_mask=predictions_mask,
                decoder_input_ids=input_ids,
                return_hypotheses=False,
            )
        else:
            raise TypeError(f"WER metric does not support decoding of type {type(self.decoding)}")

        self.tokenize = tokenize
        self.log_prediction = log_prediction
        self.batch_dim_index = batch_dim_index

    def update(
        self,
        predictions: torch.Tensor,
        predictions_lengths: torch.Tensor,
        targets: torch.Tensor,
        targets_lengths: torch.Tensor,
        predictions_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
    ):
        """
        Updates metric state.
        Args:
            predictions: an integer torch.Tensor of shape ``[Batch, Time, {Vocabulary}]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``)
            targets: an integer torch.Tensor of shape ``[Batch, Time]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``)
            target_lengths: an integer torch.Tensor of shape ``[Batch]``
            predictions_lengths: an integer torch.Tensor of shape ``[Batch]``
        """
        references = []
        with torch.no_grad():
            tgt_lenths_cpu_tensor = targets_lengths.long().cpu()
            targets_cpu_tensor = targets.long().cpu()
            # check batch_dim_index is first dim
            if self.batch_dim_index != 0:
                targets_cpu_tensor = move_dimension_to_the_front(targets_cpu_tensor, self.batch_dim_index)
            # iterate over batch
            for ind in range(targets_cpu_tensor.shape[0]):
                tgt_len = tgt_lenths_cpu_tensor[ind].item()
                target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
                reference = self.decoding.decode_tokens_to_str(target)
                references.append(reference)
            hypotheses, _ = self.decode(predictions, predictions_lengths, predictions_mask, input_ids, targets)

            if self.has_spl_tokens:
                hypotheses = [self.decoding.strip_special_tokens(hyp) for hyp in hypotheses]
                references = [self.decoding.strip_special_tokens(ref) for ref in references]

        if self.log_prediction:
            logging.info(f"\n")
            logging.info(f"reference:{references[0]}")
            logging.info(f"predicted:{hypotheses[0]}")

        super().update(hypotheses, [references])  # Note: [references] since BLEU allows multiple references.

    # Overloadng metrics for NeMo compatibility
    # TODO: Docstring
    def compute(self, return_all_metrics=True, prefix="", suffix=""):
        bleu = super().compute()
        if return_all_metrics:
            return {
                f"{prefix}bleu{suffix}": bleu,
                f"{prefix}bleu_pred_len{suffix}": self.preds_len.detach().float(),
                f"{prefix}bleu_target_len{suffix}": self.target_len.detach().float(),
                f"{prefix}bleu_num{suffix}": self.numerator.detach().float(),
                f"{prefix}bleu_denom{suffix}": self.denominator.detach().float(),
            }
        return {
            f"{prefix}bleu{suffix}": bleu,
        }

    # Adding wrapper to avoid imports and extra variables over the namespace
    def _compute_bleu(
        self, predictions_lengths, targets_lengths, numerator, denominator,
    ):
        return _bleu_score_compute(
            predictions_lengths, targets_lengths, numerator, denominator, self.n_gram, self.weights, self.smooth
        )
