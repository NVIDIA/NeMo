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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import torch

from nemo.collections.asr.modules import rnnt_abstract
from nemo.collections.asr.parts.submodules.rnnt_beam_loop_labels_computer import BeamBatchedRNNTLoopLabelsComputer
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.core.classes import typecheck
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import pack_hypotheses
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import _GreedyRNNTInfer

from omegaconf import DictConfig

class BeamTDTInferBatched(_GreedyRNNTInfer):
    """A batch level greedy TDT decoder.
    Batch level greedy decoding, performed auto-regressively.
    Args:
        decoder_model: rnnt_utils.AbstractRNNTDecoder implementation.
        joint_model: rnnt_utils.AbstractRNNTJoint implementation.
        blank_index: int index of the blank token. Must be len(vocabulary) for TDT models.
        durations: a list containing durations.
        max_symbols_per_step: Optional int. The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        preserve_alignments: Bool flag which preserves the history of alignments generated during
            greedy decoding (sample / batched). When set to true, the Hypothesis will contain
            the non-null value for `alignments` in it. Here, `alignments` is a List of List of
            Tuple(Tensor (of length V + 1 + num-big-blanks), Tensor(scalar, label after argmax)).
            The length of the list corresponds to the Acoustic Length (T).
            Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more targets from a vocabulary.
            U is the number of target tokens for the current timestep Ti.
        preserve_frame_confidence: Bool flag which preserves the history of per-frame confidence scores generated
            during greedy decoding (sample / batched). When set to true, the Hypothesis will contain
            the non-null value for `frame_confidence` in it. Here, `frame_confidence` is a List of List of floats.
            The length of the list corresponds to the Acoustic Length (T).
            Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more confidence scores.
            U is the number of target tokens for the current timestep Ti.
        include_duration_confidence: Bool flag indicating that the duration confidence scores are to be calculated and
            attached to the regular frame confidence,
            making TDT frame confidence element a pair: (`prediction_confidence`, `duration_confidence`).
        confidence_method_cfg: A dict-like object which contains the method name and settings to compute per-frame
            confidence scores.

            name: The method name (str).
                Supported values:
                    - 'max_prob' for using the maximum token probability as a confidence.
                    - 'entropy' for using a normalized entropy of a log-likelihood vector.

            entropy_type: Which type of entropy to use (str). Used if confidence_method_cfg.name is set to `entropy`.
                Supported values:
                    - 'gibbs' for the (standard) Gibbs entropy. If the alpha (α) is provided,
                        the formula is the following: H_α = -sum_i((p^α_i)*log(p^α_i)).
                        Note that for this entropy, the alpha should comply the following inequality:
                        (log(V)+2-sqrt(log^2(V)+4))/(2*log(V)) <= α <= (1+log(V-1))/log(V-1)
                        where V is the model vocabulary size.
                    - 'tsallis' for the Tsallis entropy with the Boltzmann constant one.
                        Tsallis entropy formula is the following: H_α = 1/(α-1)*(1-sum_i(p^α_i)),
                        where α is a parameter. When α == 1, it works like the Gibbs entropy.
                        More: https://en.wikipedia.org/wiki/Tsallis_entropy
                    - 'renyi' for the Rényi entropy.
                        Rényi entropy formula is the following: H_α = 1/(1-α)*log_2(sum_i(p^α_i)),
                        where α is a parameter. When α == 1, it works like the Gibbs entropy.
                        More: https://en.wikipedia.org/wiki/R%C3%A9nyi_entropy

            alpha: Power scale for logsoftmax (α for entropies). Here we restrict it to be > 0.
                When the alpha equals one, scaling is not applied to 'max_prob',
                and any entropy type behaves like the Shannon entropy: H = -sum_i(p_i*log(p_i))

            entropy_norm: A mapping of the entropy value to the interval [0,1].
                Supported values:
                    - 'lin' for using the linear mapping.
                    - 'exp' for using exponential mapping with linear shift.

        use_cuda_graph_decoder: if CUDA graphs should be enabled for decoding (currently recommended only for inference)
    """

    def __init__(
        self,
        decoder_model: rnnt_abstract.AbstractRNNTDecoder,
        joint_model: rnnt_abstract.AbstractRNNTJoint,
        blank_index: int,
        beam_size: int = 4,
        score_norm: bool = True,
        return_best_hypothesis: bool = True,
        softmax_temperature: float = 1.0,
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments: bool = False,
        preserve_frame_confidence: bool = False,
        include_duration_confidence: bool = False,
        confidence_method_cfg: Optional[DictConfig] = None,
    ):
        super().__init__(
            decoder_model=decoder_model,
            joint_model=joint_model,
            blank_index=blank_index,
            max_symbols_per_step=max_symbols_per_step,
            preserve_alignments=preserve_alignments,
            preserve_frame_confidence=preserve_frame_confidence,
            confidence_method_cfg=confidence_method_cfg,
        )

        # Depending on availability of `blank_as_pad` support
        # switch between more efficient batch decoding technique
        self._decoding_computer = None
        if self.decoder.blank_as_pad:
            # batched "loop frames" is not implemented for TDT
            self._decoding_computer = BeamBatchedRNNTLoopLabelsComputer(
                decoder=self.decoder,
                joint=self.joint,
                blank_index=self._blank_index,
                beam_size=beam_size,
                max_symbols_per_step=self.max_symbols,
                preserve_alignments=preserve_alignments,
                preserve_frame_confidence=preserve_frame_confidence,
                include_duration_confidence=include_duration_confidence,
                confidence_method_cfg=confidence_method_cfg,
            )
            self._greedy_decode = self._greedy_decode_blank_as_pad_loop_labels
        else:
            self._greedy_decode = self._greedy_decode_masked

    @typecheck()
    def forward(
        self,
        encoder_output: torch.Tensor,
        encoded_lengths: torch.Tensor,
        partial_hypotheses: Optional[List[rnnt_utils.Hypothesis]] = None,
    ):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-regressively.
        Args:
            encoder_output: A tensor of size (batch, features, timesteps).
            encoded_lengths: list of int representing the length of each sequence
                output sequence.
        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """
        # Preserve decoder and joint training state
        decoder_training_state = self.decoder.training
        joint_training_state = self.joint.training

        with torch.inference_mode():
            # Apply optional preprocessing
            encoder_output = encoder_output.transpose(1, 2)  # (B, T, D)
            logitlen = encoded_lengths

            self.decoder.eval()
            self.joint.eval()

            inseq = encoder_output  # [B, T, D]
            hypotheses = self._greedy_decode(
                inseq, logitlen, device=inseq.device, partial_hypotheses=partial_hypotheses
            )

            # Pack the hypotheses results
            packed_result = pack_hypotheses(hypotheses, logitlen)

        self.decoder.train(decoder_training_state)
        self.joint.train(joint_training_state)

        return (packed_result,)

    def _greedy_decode_masked(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
        device: torch.device,
        partial_hypotheses: Optional[List[rnnt_utils.Hypothesis]] = None,
    ):
        raise NotImplementedError("masked greedy-batched decode is not supported for TDT models.")

    @torch.inference_mode()
    def _greedy_decode_blank_as_pad_loop_labels(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
        device: torch.device,
        partial_hypotheses: Optional[list[rnnt_utils.Hypothesis]] = None,
    ) -> list[rnnt_utils.Hypothesis]:
        """
        Optimized batched greedy decoding.
        The main idea: search for next labels for the whole batch (evaluating Joint)
        and thus always evaluate prediction network with maximum possible batch size
        """
        if partial_hypotheses is not None:
            raise NotImplementedError("`partial_hypotheses` support is not implemented")

        batched_hyps, alignments, last_decoder_state = self._decoding_computer(x=x, out_len=out_len)
        hyps = rnnt_utils.batched_beam_hyps_to_hypotheses(batched_hyps, alignments, batch_size=x.shape[0])
        for hyp, state in zip(hyps, self.decoder.batch_split_states(last_decoder_state)):
            hyp.dec_state = state
        return hyps

    def disable_cuda_graphs(self):
        """Disable CUDA graphs (e.g., for decoding in training)"""
        if self._decoding_computer is not None:
            self._decoding_computer.disable_cuda_graphs()

    def maybe_enable_cuda_graphs(self):
        """Enable CUDA graphs (if allowed)"""
        if self._decoding_computer is not None:
            self._decoding_computer.maybe_enable_cuda_graphs()