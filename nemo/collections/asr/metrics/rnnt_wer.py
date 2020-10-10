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

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts import rnnt_beam_decoding as beam_decode
from nemo.collections.asr.parts import rnnt_greedy_decoding as greedy_decode
from nemo.collections.asr.parts.rnnt_utils import Hypothesis
from nemo.utils import logging

__all__ = ['RNNTDecodingWER']


class RNNTDecodingWER(TensorMetric):
    """
    This metric computes numerator and denominator for Overall Word Error Rate (WER) between prediction and reference texts.
    When doing distributed training/evaluation the result of res=WER(predictions, targets, target_lengths) calls
    will be all-reduced between all workers using SUM operations.
    Here contains two numbers res=[wer_numerator, wer_denominator]. WER=wer_numerator/wer_denominator.

    If used with PytorchLightning LightningModule, include wer_numerator and wer_denominators inside validation_step results.
    Then aggregate (sum) then at the end of validation epoch to correctly compute validation WER.

    Example:
        def validation_step(self, batch, batch_idx):
            ...
            wer_num, wer_denom = self.__wer(predictions, transcript, transcript_len)
            return {'val_loss': loss_value, 'val_wer_num': wer_num, 'val_wer_denom': wer_denom}

        def validation_epoch_end(self, outputs):
            ...
            wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
            wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
            tensorboard_logs = {'validation_loss': val_loss_mean, 'validation_avg_wer': wer_num / wer_denom}
            return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    Args:
        vocabulary: List of strings that describes the vocabulary of the dataset.
        batch_dim_index: Index of the batch dimension.
        use_cer: Whether to use Character Error Rate isntead of Word Error Rate.
        ctc_decode: Whether to use CTC decoding or not. Currently, must be set.
        log_prediction: Whether to log a single decoded sample per call.

    Returns:
        res: a torch.Tensor object with two elements: [wer_numerator, wer_denominator]. To correctly compute average
        text word error rate, compute wer=wer_numerator/wer_denominator
    """

    def __init__(
        self, decoding_cfg, decoder, joint, vocabulary, batch_dim_index=0,
    ):
        super(RNNTDecodingWER, self).__init__(name="RNNTWER")
        self.cfg = decoding_cfg
        self.batch_dim_index = batch_dim_index
        self.blank_id = len(vocabulary)
        self.labels_map = dict([(i, vocabulary[i]) for i in range(len(vocabulary))])
        self.use_cer = self.cfg.get('use_cer', False)
        self.log_prediction = self.cfg.get('log_prediction', True)

        possible_strategies = ['greedy', 'greedy_batch', 'beam', 'tsd', 'alsd']
        if self.cfg.strategy not in possible_strategies:
            raise ValueError(f"Decodin strategy must be one of {possible_strategies}")

        self.decoding2 = greedy_decode.GreedyRNNTInfer(
                decoder_model=decoder,
                joint_model=joint,
                blank_index=self.blank_id,
                max_symbols_per_step=self.cfg.greedy.get('max_symbols', None),
            )

        if self.cfg.strategy == 'greedy':
            self.decoding = greedy_decode.GreedyRNNTInfer(
                decoder_model=decoder,
                joint_model=joint,
                blank_index=self.blank_id,
                max_symbols_per_step=self.cfg.greedy.get('max_symbols', None),
            )

        elif self.cfg.strategy == 'greedy_batch':
            self.decoding = greedy_decode.GreedyBatchedRNNTInfer(
                decoder_model=decoder,
                joint_model=joint,
                blank_index=self.blank_id,
                max_symbols_per_step=self.cfg.greedy.get('max_symbols', None),
            )

        elif self.cfg.strategy == 'beam':
            self.decoding = beam_decode.BeamRNNTInfer(
                decoder_model=decoder,
                joint_model=joint,
                beam_size=self.cfg.beam.beam_size,
                search_type='default',
                score_norm=self.cfg.beam.get('score_norm', True),
            )

        elif self.cfg.strategy == 'tsd':
            self.decoding = beam_decode.BeamRNNTInfer(
                decoder_model=decoder,
                joint_model=joint,
                beam_size=self.cfg.beam.beam_size,
                search_type='tsd',
                score_norm=self.cfg.beam.get('score_norm', True),
                tsd_max_symbols_per_step=self.cfg.beam.get('tsd_max_symbols', 50),
            )

        elif self.cfg.strategy == 'alsd':
            self.decoding = beam_decode.BeamRNNTInfer(
                decoder_model=decoder,
                joint_model=joint,
                beam_size=self.cfg.beam.beam_size,
                search_type='alsd',
                score_norm=self.cfg.beam.get('score_norm', True),
                alsd_max_symmetric_expansion=self.cfg.beam.get('alsd_max_sym_expand', 2),
            )

    def rnnt_decoder_predictions_tensor(
        self, encoder_output: torch.Tensor, encoded_lengths: torch.Tensor
    ) -> List[str]:
        """
        Decodes a sequence of labels to words
        """
        hypotheses = []
        # Compute hypotheses
        with torch.no_grad():
            hypotheses_list = self.decoding(
                encoder_output=encoder_output, encoded_lengths=encoded_lengths
            )  # type: [List[Hypothesis]]

            # extract the hypotheses
            hypotheses_list = hypotheses_list[0]  # type: List[Hypothesis]

        # Drop predictions to CPU
        prediction_list = hypotheses_list

        for ind in range(len(prediction_list)):
            prediction = prediction_list[ind].y_sequence
            if type(prediction) != list:
                prediction = prediction.tolist()

            # RNN-T sample level is already preprocessed by implicit CTC decoding
            hypothesis = ''.join([self.labels_map[c] for c in prediction if c != self.blank_id])
            hypotheses.append(hypothesis)

        return hypotheses

    def forward(
        self,
        encoder_output: torch.Tensor,
        encoded_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
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
                reference = ''.join([self.labels_map[c] for c in target if c != self.blank_id])
                references.append(reference)

            hypotheses = self.rnnt_decoder_predictions_tensor(encoder_output, encoded_lengths)

        if self.log_prediction:
            logging.info(f"\n")
            logging.info(f"reference :{references[0]}")
            logging.info(f"decoded   :{hypotheses[0]}")

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
        return torch.tensor([scores, words], device=encoded_lengths.device)
