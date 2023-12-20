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

from typing import Callable, Dict, List, Literal, Optional, Sequence, Union

import torch
from torchmetrics.text import SacreBLEUScore
import editdistance

from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.modules.transformer.transformer_decoding import TransformerDecoding
from nemo.collections.asr.parts.submodules.ctc_decoding import AbstractCTCDecoding
from nemo.collections.asr.parts.submodules.rnnt_decoding import AbstractRNNTDecoding
from nemo.utils import logging

__all__ = ['BLEU']


def move_dimension_to_the_front(tensor, dim_index):
    all_dims = list(range(tensor.ndim))
    return tensor.permute(*([dim_index] + all_dims[:dim_index] + all_dims[dim_index + 1 :]))


class BLEU(SacreBLEUScore):

    full_state_update: bool = True

    def __init__(
        self,
        decoding: Union[AbstractCTCDecoding, AbstractRNNTDecoding, TransformerDecoding],
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

        self.decoding = decoding
        self.decode = self.decoding.transformer_decoder_predictions_tensor
        self.log_prediction = log_prediction
        self.batch_dim_index = batch_dim_index

    def update(
        self,
        encoder_output,
        encoder_lengths,
        targets,
        targets_lengths,
        encoder_mask=None,
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
            hypotheses, _ = self.decode(encoder_output=encoder_output, encoder_lengths=encoder_lengths)

        if self.log_prediction:
            logging.info(f"\n")
            logging.info(f"reference:{references[0]}")
            logging.info(f"predicted:{hypotheses[0]}")

        super().update(hypotheses, [references])

    def compute(self, prefix=""):
        if prefix:
            prefix = f"{prefix}_"
        bleu = super().compute()
        return {f"{prefix}bleu": bleu, 
                f"{prefix}bleu_pred": self.preds_len.detach().float(), 
                f"{prefix}bleu_target": self.target_len.detach().float(), 
                f"{prefix}bleu_num": self.numerator.detach().float(), 
                f"{prefix}bleu_denom": self.denominator.detach().float()
        }


class BLEUWER(BLEU):
    full_state_update: bool = True

    def __init__(
        self,
        decoding: Union[AbstractCTCDecoding, AbstractRNNTDecoding],
        tokenize: Literal["none", "13a", "zh", "intl", "char"] = "13a",
        n_gram: int = 4,
        lowercase: bool = False,
        use_cer: bool = False,
        fold_consecutive=False,
        weights: Optional[Sequence[float]] = None,
        smooth: bool = False,
        log_prediction: bool = True,
        batch_dim_index: int = 0,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(
            decoding=decoding,
            tokenize=tokenize,
            n_gram=n_gram,
            lowercase=lowercase,
            weights=weights,
            smooth=smooth,
            log_prediction=log_prediction,
            dist_sync_on_step=dist_sync_on_step)

        self.use_cer = use_cer
        self.fold_consecutive = fold_consecutive
        self.batch_dim_index = batch_dim_index

        self.decode = None
        if isinstance(self.decoding, AbstractRNNTDecoding):
            self.decode = self.decoding.rnnt_decoder_predictions_tensor
        elif isinstance(self.decoding, AbstractCTCDecoding):
            self.decode = lambda pred, pred_len: self.decoding.ctc_decoder_predictions_tensor(
                pred, decoder_outputs=pred_len, fold_consecutive=self.fold_consecutive
            )
        elif isinstance(self.decoding, TransformerBPEConfig):
            self.decode = self.decoding.transformer_decoder_predictions_tensor
        else:
            raise TypeError(f"BLEUWER metric does not support decoding of type {type(self.decoding)}")

        self.add_state("scores", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("words", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)

    def update(
        self,
        predictions: torch.Tensor,
        predictions_lengths: torch.Tensor,
        targets: torch.Tensor,
        targets_lengths: torch.Tensor,
    ):
        """
        Updates metric state.
        Args:
            predictions: an integer torch.Tensor of shape ``[Batch, Time, {Vocabulary}]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``)
            prediction_lengths: an integer torch.Tensor of shape ``[Batch]``
            targets: an integer torch.Tensor of shape ``[Batch, Time]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``)
            target_lengths: an integer torch.Tensor of shape ``[Batch]``
            predictions_lengths: an integer torch.Tensor of shape ``[Batch]``
        """
        words = 0
        scores = 0
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
            hypotheses, _ = self.decode(predictions, predictions_lengths)

        if self.log_prediction:
            logging.info(f"\n")
            logging.info(f"reference:{references[0]}")
            logging.info(f"predicted:{hypotheses[0]}")

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

        self.scores = torch.tensor(scores, device=self.scores.device, dtype=self.scores.dtype)
        self.words = torch.tensor(words, device=self.words.device, dtype=self.words.dtype)
    
        super().update(preds=hypotheses, targets=references)

    def reset(self):
        for attr in ["scores", "words"]:
            current_val = getattr(self, attr)
            setattr(self, attr, torch.tensor(0).detach().clone().to(current_val.device))

    def compute(self):
        scores = self.scores.detach().float()
        words = self.words.detach().float()
        return scores / words, scores, words
    
    def bleu_compute(self):
        return super().compute()

    def bleu_reset(self):
        super().reset()
    
