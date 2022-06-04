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

from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass
from typing import List, Optional, Union

import editdistance
import torch
from omegaconf import DictConfig, OmegaConf
from torchmetrics import Metric

from nemo.collections.asr.parts.submodules import ctc_greed_decoding
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis, NBestHypotheses
from nemo.utils import logging

__all__ = ['word_error_rate', 'WER', 'move_dimension_to_the_front']


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


def move_dimension_to_the_front(tensor, dim_index):
    all_dims = list(range(tensor.ndim))
    return tensor.permute(*([dim_index] + all_dims[:dim_index] + all_dims[dim_index + 1 :]))


class AbstractCTCDecoding(ABC):
    """
    Used for performing CTC auto-regressive / non-auto-regressive decoding of the logprobs.

    Args:
        decoding_cfg: A dict-like object which contains the following key-value pairs.
            strategy: str value which represents the type of decoding that can occur.
                Possible values are :
                -   greedy, greedy_batch (for greedy decoding).
                -   beam, tsd, alsd (for beam search decoding).

            compute_hypothesis_token_set: A bool flag, which determines whether to compute a list of decoded
                tokens as well as the decoded string. Default is False in order to avoid double decoding
                unless required.

            preserve_alignments: Bool flag which preserves the history of logprobs generated during
                decoding (sample / batched). When set to true, the Hypothesis will contain
                the non-null value for `logprobs` in it. Here, `logprobs` is a List of torch.Tensors.

                In order to obtain this hypothesis, please utilize `rnnt_decoder_predictions_tensor` function
                with the `return_hypotheses` flag set to True.

                The length of the list corresponds to the Acoustic Length (T).
                Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more targets from a vocabulary.
                U is the number of target tokens for the current timestep Ti.

            The config may further contain the following sub-dictionaries:
            "greedy":
                max_symbols: int, describing the maximum number of target tokens to decode per
                    timestep during greedy decoding. Setting to larger values allows longer sentences
                    to be decoded, at the cost of increased execution time.

            "beam":
                beam_size: int, defining the beam size for beam search. Must be >= 1.
                    If beam_size == 1, will perform cached greedy search. This might be slightly different
                    results compared to the greedy search above.

                score_norm: optional bool, whether to normalize the returned beam score in the hypotheses.
                    Set to True by default.

                return_best_hypothesis: optional bool, whether to return just the best hypothesis or all of the
                    hypotheses after beam search has concluded. This flag is set by default.

                tsd_max_sym_exp: optional int, determines number of symmetric expansions of the target symbols
                    per timestep of the acoustic model. Larger values will allow longer sentences to be decoded,
                    at increased cost to execution time.

                alsd_max_target_len: optional int or float, determines the potential maximum target sequence length.
                    If an integer is provided, it can decode sequences of that particular maximum length.
                    If a float is provided, it can decode sequences of int(alsd_max_target_len * seq_len),
                    where seq_len is the length of the acoustic model output (T).

                    NOTE:
                        If a float is provided, it can be greater than 1!
                        By default, a float of 2.0 is used so that a target sequence can be at most twice
                        as long as the acoustic model output length T.

                maes_num_steps: Number of adaptive steps to take. From the paper, 2 steps is generally sufficient,
                    and can be reduced to 1 to improve decoding speed while sacrificing some accuracy. int > 0.

                maes_prefix_alpha: Maximum prefix length in prefix search. Must be an integer, and is advised to keep this as 1
                    in order to reduce expensive beam search cost later. int >= 0.

                maes_expansion_beta: Maximum number of prefix expansions allowed, in addition to the beam size.
                    Effectively, the number of hypothesis = beam_size + maes_expansion_beta. Must be an int >= 0,
                    and affects the speed of inference since large values will perform large beam search in the next step.

                maes_expansion_gamma: Float pruning threshold used in the prune-by-value step when computing the expansions.
                    The default (2.3) is selected from the paper. It performs a comparison (max_log_prob - gamma <= log_prob[v])
                    where v is all vocabulary indices in the Vocab set and max_log_prob is the "most" likely token to be
                    predicted. Gamma therefore provides a margin of additional tokens which can be potential candidates for
                    expansion apart from the "most likely" candidate.
                    Lower values will reduce the number of expansions (by increasing pruning-by-value, thereby improving speed
                    but hurting accuracy). Higher values will increase the number of expansions (by reducing pruning-by-value,
                    thereby reducing speed but potentially improving accuracy). This is a hyper parameter to be experimentally
                    tuned on a validation set.

                softmax_temperature: Scales the logits of the joint prior to computing log_softmax.

        decoder: The Decoder/Prediction network module.
        joint: The Joint network module.
        blank_id: The id of the RNNT blank token.
    """

    def __init__(self, decoding_cfg, blank_id: int):
        super().__init__()

        if is_dataclass(decoding_cfg):
            decoding_cfg = OmegaConf.structured(decoding_cfg)

        if not isinstance(decoding_cfg, DictConfig):
            decoding_cfg = OmegaConf.create(decoding_cfg)

        OmegaConf.set_struct(decoding_cfg, False)

        # add minimal config
        minimal_cfg = ['greedy']
        for item in minimal_cfg:
            if item not in decoding_cfg:
                decoding_cfg[item] = OmegaConf.create({})

        self.cfg = decoding_cfg
        self.blank_id = blank_id
        self.preserve_alignments = self.cfg.get('preserve_alignments', None)
        self.compute_timestamps = self.cfg.get('compute_timestamps', None)
        self.batch_dim_index = self.cfg.get('batch_dim_index', 0)

        possible_strategies = ['greedy']
        if self.cfg.strategy not in possible_strategies:
            raise ValueError(f"Decoding strategy must be one of {possible_strategies}. Given {self.cfg.strategy}")

        # Update preserve alignments
        if self.preserve_alignments is None:
            if self.cfg.strategy in ['greedy']:
                self.preserve_alignments = self.cfg.greedy.get('preserve_alignments', False)

        # Update compute timestamps
        if self.compute_timestamps is None:
            if self.cfg.strategy in ['greedy']:
                self.compute_timestamps = self.cfg.greedy.get('compute_timestamps', False)

        if self.cfg.strategy == 'greedy':

            self.decoding = ctc_greed_decoding.GreedyCTCInfer(
                blank_id=self.blank_id,
                preserve_alignments=self.preserve_alignments,
                compute_timestamps=self.compute_timestamps,
            )

        else:
            raise ValueError(
                f"Incorrect decoding strategy supplied. Must be one of {possible_strategies}\n"
                f"but was provided {self.cfg.strategy}"
            )

    def ctc_decoder_predictions_tensor(
        self,
        decoder_outputs: torch.Tensor,
        decoder_lengths: torch.Tensor = None,
        fold_consecutive: bool = True,
        return_hypotheses: bool = False,
    ) -> (List[str], Optional[List[List[str]]], Optional[Union[Hypothesis, NBestHypotheses]]):
        """
        Decodes a sequence of labels to words

        Args:
            decoder_outputs: An integer torch.Tensor of shape [Batch, Time, Vocabulary] (if ``batch_index_dim == 0``) or [Time, Batch]
                (if ``batch_index_dim == 1``) of integer indices that correspond to the index of some character in the
                label set.
            decoder_lengths: Optional tensor of length `Batch` which contains the integer lengths
                of the sequence in the padded `predictions` tensor.
            fold_consecutive: Bool, determine whether to perform "ctc collapse", folding consecutive tokens
                into a single token.
            return_hypotheses: Bool flag whether to return just the decoding predictions of the model
                or a Hypothesis object that holds information such as the decoded `text`,
                the `alignment` of emited by the CTC Model, and the `length` of the sequence (if available).
                May also contain the log-probabilities of the decoder (if this method is called via
                transcribe())

        Returns:
            Either a list of str which represent the CTC decoded strings per sample,
            or a list of Hypothesis objects containing additional information.
        """
        decoder_outputs = move_dimension_to_the_front(decoder_outputs, self.batch_dim_index)

        with torch.inference_mode():
            # Resolve the forward step of the decoding strategy
            hypotheses_list = self.decoding(
                decoder_output=decoder_outputs, decoder_lengths=decoder_lengths
            )  # type: List[List[Hypothesis]]

            # extract the hypotheses
            hypotheses_list = hypotheses_list[0]  # type: List[Hypothesis]

        if isinstance(hypotheses_list[0], NBestHypotheses):
            hypotheses = []
            all_hypotheses = []

            for nbest_hyp in hypotheses_list:  # type: NBestHypotheses
                n_hyps = nbest_hyp.n_best_hypotheses  # Extract all hypotheses for this sample
                decoded_hyps = self.decode_hypothesis(
                    n_hyps, fold_consecutive
                )  # type: List[Union[Hypothesis, NBestHypotheses]]
                hypotheses.append(decoded_hyps[0])  # best hypothesis
                all_hypotheses.append(decoded_hyps)

            if return_hypotheses:
                return hypotheses, all_hypotheses

            best_hyp_text = [h.text for h in hypotheses]
            all_hyp_text = [h.text for hh in all_hypotheses for h in hh]
            return best_hyp_text, all_hyp_text

        else:
            hypotheses = self.decode_hypothesis(
                hypotheses_list, fold_consecutive
            )  # type: List[Union[Hypothesis, NBestHypotheses]]

            if return_hypotheses:
                return hypotheses, None
            best_hyp_text = [h.text for h in hypotheses]
            return best_hyp_text, None

    def decode_hypothesis(
        self, hypotheses_list: List[Hypothesis], fold_consecutive: bool
    ) -> List[Union[Hypothesis, NBestHypotheses]]:
        """
        Decode a list of hypotheses into a list of strings.

        Args:
            hypotheses_list: List of Hypothesis.
            fold_consecutive:

        Returns:
            A list of strings.
        """
        for ind in range(len(hypotheses_list)):
            # Extract the integer encoded hypothesis
            hyp = hypotheses_list[ind]
            prediction = hyp.y_sequence
            predictions_len = hyp.length if hyp.length > 0 else None

            if fold_consecutive:
                if type(prediction) != list:
                    prediction = prediction.numpy().tolist()

                if predictions_len is not None:
                    prediction = prediction[:predictions_len]

                # CTC decoding procedure
                decoded_prediction = []
                previous = self.blank_id
                for p in prediction:
                    if (p != previous or previous == self.blank_id) and p != self.blank_id:
                        decoded_prediction.append(p)
                    previous = p

            else:
                if predictions_len is not None:
                    prediction = prediction[:predictions_len]
                decoded_prediction = prediction[prediction != self.blank_id].tolist()

            # De-tokenize the integer tokens
            hypothesis = self.decode_tokens_to_str(decoded_prediction)
            hypotheses_list[ind].text = hypothesis

        return hypotheses_list

    @abstractmethod
    def decode_tokens_to_str(self, tokens: List[int]) -> str:
        """
        Implemented by subclass in order to decoder a token id list into a string.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A decoded string.
        """
        raise NotImplementedError()

    @abstractmethod
    def decode_ids_to_tokens(self, tokens: List[int]) -> List[str]:
        """
        Implemented by subclass in order to decode a token id list into a token list.
        A token list is the string representation of each token id.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A list of decoded tokens.
        """
        raise NotImplementedError()

    def compute_ctc_timestamps(
        self, decoder_outputs: torch.Tensor, decoder_lengths: torch.Tensor = None, timestamp_type: str = "all"
    ):
        assert timestamp_type in ['char', 'word', 'subword', 'all']
        original_timestamps = self.compute_timestamps

        hypothesis, _ = self.ctc_decoder_predictions_tensor(decoder_outputs, decoder_lengths, return_hypotheses=True)

        # reset timestamps
        self.compute_timestamps = original_timestamps

    @property
    def preserve_alignments(self):
        return self._preserve_alignments

    @preserve_alignments.setter
    def preserve_alignments(self, value):
        self._preserve_alignments = value

        if hasattr(self, 'decoding'):
            self.decoding.preserve_alignments = value

    @property
    def compute_timestamps(self):
        return self._compute_timestamps

    @compute_timestamps.setter
    def compute_timestamps(self, value):
        self._compute_timestamps = value

        if hasattr(self, 'decoding'):
            self.decoding.compute_timestamps = value


class CTCCharDecoding(AbstractCTCDecoding):
    """
    Used for performing CTC auto-regressive / non-auto-regressive decoding of the logprobs.

    Args:
        decoding_cfg: A dict-like object which contains the following key-value pairs.
            strategy: str value which represents the type of decoding that can occur.
                Possible values are :
                -   greedy, greedy_batch (for greedy decoding).
                -   beam, tsd, alsd (for beam search decoding).

            compute_hypothesis_token_set: A bool flag, which determines whether to compute a list of decoded
                tokens as well as the decoded string. Default is False in order to avoid double decoding
                unless required.

            preserve_alignments: Bool flag which preserves the history of logprobs generated during
                decoding (sample / batched). When set to true, the Hypothesis will contain
                the non-null value for `logprobs` in it. Here, `logprobs` is a List of torch.Tensors.

                In order to obtain this hypothesis, please utilize `rnnt_decoder_predictions_tensor` function
                with the `return_hypotheses` flag set to True.

                The length of the list corresponds to the Acoustic Length (T).
                Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more targets from a vocabulary.
                U is the number of target tokens for the current timestep Ti.

            The config may further contain the following sub-dictionaries:
            "greedy":
                max_symbols: int, describing the maximum number of target tokens to decode per
                    timestep during greedy decoding. Setting to larger values allows longer sentences
                    to be decoded, at the cost of increased execution time.

            "beam":
                beam_size: int, defining the beam size for beam search. Must be >= 1.
                    If beam_size == 1, will perform cached greedy search. This might be slightly different
                    results compared to the greedy search above.

                score_norm: optional bool, whether to normalize the returned beam score in the hypotheses.
                    Set to True by default.

                return_best_hypothesis: optional bool, whether to return just the best hypothesis or all of the
                    hypotheses after beam search has concluded. This flag is set by default.

                tsd_max_sym_exp: optional int, determines number of symmetric expansions of the target symbols
                    per timestep of the acoustic model. Larger values will allow longer sentences to be decoded,
                    at increased cost to execution time.

                alsd_max_target_len: optional int or float, determines the potential maximum target sequence length.
                    If an integer is provided, it can decode sequences of that particular maximum length.
                    If a float is provided, it can decode sequences of int(alsd_max_target_len * seq_len),
                    where seq_len is the length of the acoustic model output (T).

                    NOTE:
                        If a float is provided, it can be greater than 1!
                        By default, a float of 2.0 is used so that a target sequence can be at most twice
                        as long as the acoustic model output length T.

                                maes_num_steps: Number of adaptive steps to take. From the paper, 2 steps is generally sufficient,
                    and can be reduced to 1 to improve decoding speed while sacrificing some accuracy. int > 0.

                maes_prefix_alpha: Maximum prefix length in prefix search. Must be an integer, and is advised to keep this as 1
                    in order to reduce expensive beam search cost later. int >= 0.

                maes_expansion_beta: Maximum number of prefix expansions allowed, in addition to the beam size.
                    Effectively, the number of hypothesis = beam_size + maes_expansion_beta. Must be an int >= 0,
                    and affects the speed of inference since large values will perform large beam search in the next step.

                maes_expansion_gamma: Float pruning threshold used in the prune-by-value step when computing the expansions.
                    The default (2.3) is selected from the paper. It performs a comparison (max_log_prob - gamma <= log_prob[v])
                    where v is all vocabulary indices in the Vocab set and max_log_prob is the "most" likely token to be
                    predicted. Gamma therefore provides a margin of additional tokens which can be potential candidates for
                    expansion apart from the "most likely" candidate.
                    Lower values will reduce the number of expansions (by increasing pruning-by-value, thereby improving speed
                    but hurting accuracy). Higher values will increase the number of expansions (by reducing pruning-by-value,
                    thereby reducing speed but potentially improving accuracy). This is a hyper parameter to be experimentally
                    tuned on a validation set.

                softmax_temperature: Scales the logits of the joint prior to computing log_softmax.

        decoder: The Decoder/Prediction network module.
        joint: The Joint network module.
        vocabulary: The vocabulary (excluding the RNNT blank token) which will be used for decoding.
    """

    def __init__(
        self, decoding_cfg, vocabulary,
    ):
        blank_id = len(vocabulary)
        self.vocabulary = vocabulary
        self.labels_map = dict([(i, vocabulary[i]) for i in range(len(vocabulary))])

        super().__init__(decoding_cfg=decoding_cfg, blank_id=blank_id)

    def decode_tokens_to_str(self, tokens: List[int]) -> str:
        """
        Implemented by subclass in order to decoder a token list into a string.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A decoded string.
        """
        hypothesis = ''.join(self.decode_ids_to_tokens(tokens))
        return hypothesis

    def decode_ids_to_tokens(self, tokens: List[int]) -> List[str]:
        """
        Implemented by subclass in order to decode a token id list into a token list.
        A token list is the string representation of each token id.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A list of decoded tokens.
        """
        token_list = [self.labels_map[c] for c in tokens if c != self.blank_id]
        return token_list


class WER(Metric):
    """
    This metric computes numerator and denominator for Overall Word Error Rate (WER) between prediction and reference
    texts. When doing distributed training/evaluation the result of ``res=WER(predictions, targets, target_lengths)``
    calls will be all-reduced between all workers using SUM operations. Here ``res`` contains three numbers
    ``res=[wer, total_levenstein_distance, total_number_of_words]``.

    If used with PytorchLightning LightningModule, include wer_numerator and wer_denominators inside validation_step
    results. Then aggregate (sum) then at the end of validation epoch to correctly compute validation WER.

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
        batch_dim_index: Index of the batch dimension of ``targets`` and ``predictions`` parameters of ``__call__``,
            ``forward``, ``update``, ``ctc_decoder_predictions_tensor`` methods. Can be either 0 or 1.
        use_cer: Whether to use Character Error Rate instead of Word Error Rate.
        ctc_decode: Whether to use CTC decoding or not. Currently, must be set.
        log_prediction: Whether to log a single decoded sample per call.
        fold_consecutive: Whether repeated consecutive characters should be folded into one when decoding.

    Returns:
        res: a tuple of 3 zero dimensional float32 ``torch.Tensor` objects: a WER score, a sum of Levenstein's
            distances for all prediction - reference pairs, total number of words in all references.
    """

    full_state_update: bool = True

    def __init__(
        self,
        decoding: CTCCharDecoding,
        use_cer=False,
        log_prediction=True,
        fold_consecutive=True,
        dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)

        self.decoding = decoding
        self.use_cer = use_cer
        self.log_prediction = log_prediction
        self.fold_consecutive = fold_consecutive

        self.add_state("scores", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("words", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        predictions_lengths: torch.Tensor = None,
    ):
        """
        Updates metric state.
        Args:
            predictions: an integer torch.Tensor of shape ``[Batch, Time, Vocabulary]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``)
            targets: an integer torch.Tensor of shape ``[Batch, Time]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``)
            target_lengths: an integer torch.Tensor of shape ``[Batch]``
            predictions_lengths: an integer torch.Tensor of shape ``[Batch]``
        """
        words = 0.0
        scores = 0.0
        references = []
        with torch.no_grad():
            # prediction_cpu_tensor = tensors[0].long().cpu()
            targets_cpu_tensor = targets.long().cpu()
            tgt_lenths_cpu_tensor = target_lengths.long().cpu()

            # iterate over batch
            for ind in range(targets_cpu_tensor.shape[0]):
                tgt_len = tgt_lenths_cpu_tensor[ind].item()
                target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
                reference = self.decoding.decode_tokens_to_str(target)
                references.append(reference)

            hypotheses, _ = self.decoding.ctc_decoder_predictions_tensor(
                predictions, predictions_lengths, fold_consecutive=self.fold_consecutive
            )

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
        # return torch.tensor([scores, words]).to(predictions.device)

    def compute(self):
        scores = self.scores.detach().float()
        words = self.words.detach().float()
        return scores / words, scores, words


@dataclass
class CTCCharDecodingConfig:
    strategy: str = "greedy"

    # preserve decoding alignments
    preserve_alignments: Optional[bool] = None

    # compute ctc time stamps
    compute_timestamps: Optional[bool] = None

    # batch dimension
    batch_dim_index: int = 0

    # greedy decoding config
    greedy: ctc_greed_decoding.GreedyCTCInferConfig = ctc_greed_decoding.GreedyCTCInferConfig()
