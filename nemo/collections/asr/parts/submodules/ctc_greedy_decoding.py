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

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.asr.parts.context_biasing.gpu_boosting.boosting_graph_batched import GPUBoostingTreeModel
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodConfig, ConfidenceMethodMixin
from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import HypothesisType, LengthsType, LogprobsType, NeuralType
from nemo.core.utils.cuda_python_utils import (
    check_cuda_python_cuda_graphs_conditional_nodes_supported,
    cu_call,
    run_nvrtc,
    with_conditional_node,
)
from nemo.utils import logging, logging_mode

try:
    from cuda import cudart

    HAVE_CUDA_PYTHON = True
except ImportError:
    HAVE_CUDA_PYTHON = False

NEG_INF = float("-inf")


from nemo.core.utils.optional_libs import TRITON_AVAILABLE


class CTCDecoderCudaGraphsState:
    """
    State for Loop Labels algorithm. Used only with CUDA graphs.
    In initialization phase it is possible to assign values (tensors) to the state.
    For algorithm code the storage should be reused (prefer copy data instead of assigning tensors).
    """

    max_time: int  # maximum length of internal storage for time dimension
    batch_size: int  # (maximum) length of internal storage for batch dimension
    device: torch.device  # device to store preallocated tensors
    float_dtype: torch.dtype

    frame_i: torch.Tensor
    decoding_active: torch.Tensor

    logits: torch.Tensor  # projected output from the encoder for decoding algorithm
    logits_len: torch.Tensor  # length of the (projected) output from the encoder

    labels: torch.Tensor  # storage for current labels
    last_labels: torch.Tensor  # storage for previous labels
    scores: torch.Tensor  # storage for current scores

    batch_indices: torch.Tensor  # indices of elements in batch (constant, range [0, batch_size-1])

    batch_lm_states: Optional[torch.Tensor] = None
    lm_scores: Optional[torch.Tensor] = None
    batch_lm_states_candidates: Optional[torch.Tensor] = None

    prediction_labels: torch.Tensor
    prediction_logprobs: torch.Tensor

    full_graph = None

    def __init__(
        self,
        batch_size: int,
        max_time: int,
        encoder_dim: int,
        device: torch.device,
        float_dtype: torch.dtype,
    ):
        """

        Args:
            batch_size: batch size for encoder output storage
            max_time: maximum time for encoder output storage
            encoder_dim: last dimension for encoder output storage (projected encoder output)
            max_symbols: max symbols per step (to avoid infinite looping and pre-allocate storage)
            device: device to store tensors
            float_dtype: default float dtype for tensors (should match projected encoder output)
        """
        self.device = device
        self.float_dtype = float_dtype
        self.batch_size = batch_size
        self.max_time = max_time

        self.frame_i = torch.tensor(0, dtype=torch.long, device=device)
        self.decoding_active = torch.tensor(True, dtype=torch.bool, device=device)

        self.logits = torch.zeros(
            (self.batch_size, self.max_time, encoder_dim),
            dtype=float_dtype,
            device=self.device,
        )
        self.logits_len = torch.zeros((self.batch_size,), dtype=torch.long, device=self.device)

        self.labels = torch.zeros([self.batch_size], dtype=torch.long, device=self.device)
        self.last_labels = torch.zeros([self.batch_size], dtype=torch.long, device=self.device)
        self.scores = torch.zeros([self.batch_size], dtype=float_dtype, device=self.device)

        # indices of elements in batch (constant)
        self.batch_indices = torch.arange(self.batch_size, dtype=torch.long, device=self.device)

        self.batch_lm_states = torch.zeros([batch_size], dtype=torch.long, device=device)
        self.batch_lm_states_candidates = torch.zeros([batch_size, encoder_dim], dtype=torch.long, device=device)
        self.lm_scores = torch.zeros([batch_size, encoder_dim], dtype=float_dtype, device=device)

        self.predictions_labels = torch.zeros([batch_size, max_time], device=device, dtype=torch.long)
        self.predictions_logprobs = torch.zeros([batch_size, max_time], device=device, dtype=float_dtype)

    def need_reinit(self, logits: torch.Tensor) -> bool:
        """Check if need to reinit state: larger batch_size/max_time, or new device"""
        return (
            self.batch_size < logits.shape[0]
            or self.max_time < logits.shape[1]
            or self.device.index != logits.device.index
        )


if TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    from nemo.collections.asr.parts.submodules.ngram_lm.ngram_lm_triton import ngram_advance_triton_kernel

    @triton.jit
    def _ctc_greedy_decode_lm_triton(
        logits_ptr,
        out_len_ptr,
        vocab_size: "tl.constexpr",
        states_ptr,
        new_states_ptr,
        scores_ptr,
        start_state: int,
        max_order: int,
        backoff_to_states_ptr,
        backoff_weights_ptr,
        state_start_arcs_ptr,
        state_end_arcs_ptr,
        to_states_ptr,
        ilabels_ptr,
        arcs_weights_ptr,
        BLOCK_SIZE: "tl.constexpr",
    ):
        ngram_advance_triton_kernel(
            vocab_size=vocab_size,
            states_ptr=states_ptr,
            new_states_ptr=new_states_ptr,
            scores_ptr=scores_ptr,
            start_state=start_state,
            max_order=max_order,
            backoff_to_states_ptr=backoff_to_states_ptr,
            backoff_weights_pt=backoff_weights_ptr,
            state_start_arcs_ptr=state_start_arcs_ptr,
            state_end_arcs_ptr=state_end_arcs_ptr,
            to_states_ptr=to_states_ptr,
            ilabels_ptr=ilabels_ptr,
            arcs_weights_ptr=arcs_weights_ptr,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        ...


def pack_hypotheses(
    hypotheses: List[rnnt_utils.Hypothesis],
    logitlen: torch.Tensor,
) -> List[rnnt_utils.Hypothesis]:

    if logitlen is not None:
        if hasattr(logitlen, 'cpu'):
            logitlen_cpu = logitlen.to('cpu')
        else:
            logitlen_cpu = logitlen

    for idx, hyp in enumerate(hypotheses):  # type: rnnt_utils.Hypothesis
        hyp.y_sequence = torch.tensor(hyp.y_sequence, dtype=torch.long)

        if logitlen is not None:
            hyp.length = logitlen_cpu[idx]

        if hyp.dec_state is not None:
            hyp.dec_state = _states_to_device(hyp.dec_state)

    return hypotheses


def _states_to_device(dec_state, device='cpu'):
    if torch.is_tensor(dec_state):
        dec_state = dec_state.to(device)

    elif isinstance(dec_state, (list, tuple)):
        dec_state = tuple(_states_to_device(dec_i, device) for dec_i in dec_state)

    return dec_state


_DECODER_LENGTHS_NONE_WARNING = "Passing in decoder_lengths=None for CTC decoding is likely to be an error, since it is unlikely that each element of your batch has exactly the same length. decoder_lengths will default to decoder_output.shape[0]."


class GreedyCTCInfer(Typing, ConfidenceMethodMixin):
    """A greedy CTC decoder.

    Provides a common abstraction for sample level and batch level greedy decoding.

    Args:
        blank_index: int index of the blank token. Can be 0 or len(vocabulary).
        preserve_alignments: Bool flag which preserves the history of logprobs generated during
            decoding (sample / batched). When set to true, the Hypothesis will contain
            the non-null value for `logprobs` in it. Here, `logprobs` is a torch.Tensors.
        compute_timestamps: A bool flag, which determines whether to compute the character/subword, or
                word based timestamp mapping the output log-probabilities to discrite intervals of timestamps.
                The timestamps will be available in the returned Hypothesis.timestep as a dictionary.
        preserve_frame_confidence: Bool flag which preserves the history of per-frame confidence scores
            generated during decoding. When set to true, the Hypothesis will contain
            the non-null value for `frame_confidence` in it. Here, `frame_confidence` is a List of floats.
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

    """

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        # Input can be of dimension -
        # ('B', 'T', 'D') [Log probs] or ('B', 'T') [Labels]

        return {
            "decoder_output": NeuralType(None, LogprobsType()),
            "decoder_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return {"predictions": [NeuralType(elements_type=HypothesisType())]}

    def __init__(
        self,
        blank_id: int,
        preserve_alignments: bool = False,
        compute_timestamps: bool = False,
        preserve_frame_confidence: bool = False,
        confidence_method_cfg: Optional[DictConfig] = None,
    ):
        super().__init__()

        self.blank_id = blank_id
        self.preserve_alignments = preserve_alignments
        # we need timestamps to extract non-blank per-frame confidence
        self.compute_timestamps = compute_timestamps | preserve_frame_confidence
        self.preserve_frame_confidence = preserve_frame_confidence

        # set confidence calculation method
        self._init_confidence_method(confidence_method_cfg)

    @typecheck()
    def forward(
        self,
        decoder_output: torch.Tensor,
        decoder_lengths: Optional[torch.Tensor],
    ):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-repressively.

        Args:
            decoder_output: A tensor of size (batch, timesteps, features) or (batch, timesteps) (each timestep is a label).
            decoder_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """

        logging.warning(
            "CTC decoding strategy 'greedy' is slower than 'greedy_batch', which implements the same exact interface. Consider changing your strategy to 'greedy_batch' for a free performance improvement.",
            mode=logging_mode.ONCE,
        )

        if decoder_lengths is None:
            logging.warning(_DECODER_LENGTHS_NONE_WARNING, mode=logging_mode.ONCE)

        with torch.inference_mode():
            hypotheses = []
            # Process each sequence independently

            if decoder_output.is_cuda:
                # This two-liner is around twenty times faster than:
                # `prediction_cpu_tensor = decoder_output.cpu()`
                # cpu() does not use pinned memory, meaning that a slow pageable
                # copy must be done instead.
                prediction_cpu_tensor = torch.empty(
                    decoder_output.shape, dtype=decoder_output.dtype, device=torch.device("cpu"), pin_memory=True
                )
                prediction_cpu_tensor.copy_(decoder_output, non_blocking=True)
            else:
                prediction_cpu_tensor = decoder_output

            if decoder_lengths is not None and isinstance(decoder_lengths, torch.Tensor):
                # Before this change, self._greedy_decode_labels would copy
                # each scalar from GPU to CPU one at a time, in the line:
                # prediction = prediction[:out_len]
                # Doing one GPU to CPU copy ahead of time amortizes that overhead.
                decoder_lengths = decoder_lengths.cpu()

            if prediction_cpu_tensor.ndim < 2 or prediction_cpu_tensor.ndim > 3:
                raise ValueError(
                    f"`decoder_output` must be a tensor of shape [B, T] (labels, int) or "
                    f"[B, T, V] (log probs, float). Provided shape = {prediction_cpu_tensor.shape}"
                )

            # determine type of input - logprobs or labels
            if prediction_cpu_tensor.ndim == 2:  # labels
                greedy_decode = self._greedy_decode_labels
            else:
                greedy_decode = self._greedy_decode_logprobs

            for ind in range(prediction_cpu_tensor.shape[0]):
                out_len = decoder_lengths[ind] if decoder_lengths is not None else None
                hypothesis = greedy_decode(prediction_cpu_tensor[ind], out_len)
                hypotheses.append(hypothesis)

            # Pack results into Hypotheses
            packed_result = pack_hypotheses(hypotheses, decoder_lengths)

        return (packed_result,)

    @torch.no_grad()
    def _greedy_decode_logprobs(self, x: torch.Tensor, out_len: Optional[torch.Tensor]):
        # x: [T, D]
        # out_len: [seq_len]

        # Initialize blank state and empty label set in Hypothesis
        hypothesis = rnnt_utils.Hypothesis(score=0.0, y_sequence=[], dec_state=None, timestamp=[], last_token=None)
        prediction = x.cpu()

        if out_len is not None:
            prediction = prediction[:out_len]

        prediction_logprobs, prediction_labels = prediction.max(dim=-1)

        non_blank_ids = prediction_labels != self.blank_id
        hypothesis.y_sequence = prediction_labels.tolist()
        hypothesis.score = (prediction_logprobs[non_blank_ids]).sum()

        if self.preserve_alignments:
            # Preserve the logprobs, as well as labels after argmax
            hypothesis.alignments = (prediction.clone(), prediction_labels.clone())

        if self.compute_timestamps:
            hypothesis.timestamp = torch.nonzero(non_blank_ids, as_tuple=False)[:, 0].tolist()

        if self.preserve_frame_confidence:
            hypothesis.frame_confidence = self._get_confidence(prediction)

        return hypothesis

    @torch.no_grad()
    def _greedy_decode_labels(self, x: torch.Tensor, out_len: Optional[torch.Tensor]):
        # x: [T]
        # out_len: [seq_len]

        # Initialize blank state and empty label set in Hypothesis
        hypothesis = rnnt_utils.Hypothesis(score=0.0, y_sequence=[], dec_state=None, timestamp=[], last_token=None)
        prediction_labels = x.cpu()

        if out_len is not None:
            prediction_labels = prediction_labels[:out_len]

        non_blank_ids = prediction_labels != self.blank_id
        hypothesis.y_sequence = prediction_labels.tolist()
        hypothesis.score = -1.0

        if self.preserve_alignments:
            raise ValueError("Requested for alignments, but predictions provided were labels, not log probabilities.")

        if self.compute_timestamps:
            hypothesis.timestamp = torch.nonzero(non_blank_ids, as_tuple=False)[:, 0].tolist()

        if self.preserve_frame_confidence:
            raise ValueError(
                "Requested for per-frame confidence, but predictions provided were labels, not log probabilities."
            )

        return hypothesis

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class GreedyBatchedCTCInfer(Typing, ConfidenceMethodMixin):
    """A vectorized greedy CTC decoder.

    This is basically always faster than GreedyCTCInfer, and supports
    the same interface. See issue #8891 on github for what is wrong
    with GreedyCTCInfer. GreedyCTCInfer loops over each element in the
    batch, running kernels at batch size one. CPU overheads end up
    dominating. This implementation does appropriate masking to
    appropriately do the same operation in a batched manner.

    Args:
        blank_index: int index of the blank token. Can be 0 or len(vocabulary).
        preserve_alignments: Bool flag which preserves the history of logprobs generated during
            decoding (sample / batched). When set to true, the Hypothesis will contain
            the non-null value for `logprobs` in it. Here, `logprobs` is a torch.Tensors.
        compute_timestamps: A bool flag, which determines whether to compute the character/subword, or
                word based timestamp mapping the output log-probabilities to discrite intervals of timestamps.
                The timestamps will be available in the returned Hypothesis.timestep as a dictionary.
        preserve_frame_confidence: Bool flag which preserves the history of per-frame confidence scores
            generated during decoding. When set to true, the Hypothesis will contain
            the non-null value for `frame_confidence` in it. Here, `frame_confidence` is a List of floats.
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

    """

    ngram_lm_batch: Optional[NGramGPULanguageModel]

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        # Input can be of dimension -
        # ('B', 'T', 'D') [Log probs] or ('B', 'T') [Labels]

        return {
            "decoder_output": NeuralType(None, LogprobsType()),
            "decoder_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return {"predictions": [NeuralType(elements_type=HypothesisType())]}

    def __init__(
        self,
        blank_id: int,
        preserve_alignments: bool = False,
        compute_timestamps: bool = False,
        preserve_frame_confidence: bool = False,
        confidence_method_cfg: Optional[DictConfig] = None,
        ngram_lm_model: Optional[str | Path] = None,
        ngram_lm_alpha: float = 0.0,
        btree_model: Optional[str | Path] = None,
        btree_alpha: float = 0.0,
    ):
        super().__init__()

        self.blank_id = blank_id
        self.preserve_alignments = preserve_alignments
        # we need timestamps to extract non-blank per-frame confidence
        self.compute_timestamps = compute_timestamps | preserve_frame_confidence
        self.preserve_frame_confidence = preserve_frame_confidence

        # set confidence calculation method
        self._init_confidence_method(confidence_method_cfg)

        # init ngram lm
        if ngram_lm_model is not None:
            self.ngram_lm_batch = NGramGPULanguageModel.from_file(lm_path=ngram_lm_model, vocab_size=self.blank_id)
        else:
            self.ngram_lm_batch = None
            
        # init ngram lm
        if btree_model is not None:
            self.wb_batch = GPUBoostingTreeModel.from_file(lm_path=btree_model, vocab_size=self.blank_id)
        else:
            self.wb_batch = None
        self.ngram_lm_alpha = ngram_lm_alpha
        self.btree_alpha = btree_alpha
        self._cuda_graphs_state: CTCDecoderCudaGraphsState | None = None

    @typecheck()
    def forward(
        self,
        decoder_output: torch.Tensor,
        decoder_lengths: Optional[torch.Tensor],
    ):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-repressively.

        Args:
            decoder_output: A tensor of size (batch, timesteps, features) or (batch, timesteps) (each timestep is a label).
            decoder_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """

        input_decoder_lengths = decoder_lengths

        if decoder_lengths is None:
            logging.warning(_DECODER_LENGTHS_NONE_WARNING, mode=logging_mode.ONCE)
            decoder_lengths = torch.tensor(
                [decoder_output.shape[1]], dtype=torch.long, device=decoder_output.device
            ).expand(decoder_output.shape[0])

        # GreedyCTCInfer::forward(), by accident, works with
        # decoder_lengths on either CPU or GPU when decoder_output is
        # on GPU. For the sake of backwards compatibility, we also
        # allow decoder_lengths to be on the CPU device. In this case,
        # we simply copy the decoder_lengths from CPU to GPU. If both
        # tensors are already on the same device, this is a no-op.
        decoder_lengths = decoder_lengths.to(decoder_output.device)

        if decoder_output.ndim == 2:
            if self.ngram_lm_batch is not None:
                raise NotImplementedError
            if self.wb_batch is not None:
                raise NotImplementedError
            hypotheses = self._greedy_decode_labels_batched(decoder_output, decoder_lengths)
        else:
            if self.ngram_lm_batch is None and self.wb_batch is None:
                hypotheses = self._greedy_decode_logprobs_batched(decoder_output, decoder_lengths)
            else:
                if self.ngram_lm_batch is not None:
                    self.ngram_lm_batch.to(decoder_output.device)
                if self.wb_batch is not None:
                    self.wb_batch.to(decoder_output.device)
                hypotheses = self._greedy_decode_logprobs_batched_lm(decoder_output, decoder_lengths)
        packed_result = pack_hypotheses(hypotheses, input_decoder_lengths)
        return (packed_result,)

    @torch.no_grad()
    def _greedy_decode_logprobs_batched(self, x: torch.Tensor, out_len: torch.Tensor):
        # x: [B, T, D]
        # out_len: [B]

        batch_size = x.shape[0]
        max_time = x.shape[1]

        predictions = x
        # In CTC greedy decoding, each output maximum likelihood token
        # is calculated independent of the other tokens.
        predictions_logprobs, predictions_labels = predictions.max(dim=-1)

        # Since predictions_logprobs is a padded matrix in the time
        # dimension, we consider invalid timesteps to be "blank".
        time_steps = torch.arange(max_time, device=x.device).unsqueeze(0).expand(batch_size, max_time)
        non_blank_ids_mask = torch.logical_and(predictions_labels != self.blank_id, time_steps < out_len.unsqueeze(1))
        # Sum the non-blank labels to compute the score of the
        # transcription. This follows from Eq. (3) of "Connectionist
        # Temporal Classification: Labelling Unsegmented Sequence Data
        # with Recurrent Neural Networks".
        scores = torch.where(non_blank_ids_mask, predictions_logprobs, 0.0).sum(axis=1)

        scores = scores.cpu()
        predictions_labels = predictions_labels.cpu()
        out_len = out_len.cpu()

        if self.preserve_alignments or self.preserve_frame_confidence:
            predictions = predictions.cpu()

        hypotheses = []

        # This mimics the for loop in GreedyCTCInfer::forward.
        for i in range(batch_size):
            hypothesis = rnnt_utils.Hypothesis(score=0.0, y_sequence=[], dec_state=None, timestamp=[], last_token=None)
            hypothesis.score = scores[i]

            prediction_labels_no_padding = predictions_labels[i, : out_len[i]].tolist()

            assert predictions_labels.dtype == torch.int64
            hypothesis.y_sequence = prediction_labels_no_padding

            if self.preserve_alignments:
                hypothesis.alignments = (
                    predictions[i, : out_len[i], :].clone(),
                    predictions_labels[i, : out_len[i]].clone(),
                )
            if self.compute_timestamps:
                # TOOD: Could do this in a vectorized manner... Would
                # prefer to have nonzero_static, though, for sanity.
                # Or do a prefix sum on out_len
                hypothesis.timestamp = torch.nonzero(non_blank_ids_mask[i], as_tuple=False)[:, 0].cpu().tolist()
            if self.preserve_frame_confidence:
                hypothesis.frame_confidence = self._get_confidence(predictions[i, : out_len[i], :])

            hypotheses.append(hypothesis)

        return hypotheses

    @torch.no_grad()
    def _greedy_decode_labels_batched(self, x: torch.Tensor, out_len: torch.Tensor):
        """
        This does greedy decoding in the case where you have already found the
        most likely token at each timestep.
        """
        # x: [B, T]
        # out_len: [B]

        batch_size = x.shape[0]
        max_time = x.shape[1]

        predictions_labels = x
        time_steps = torch.arange(max_time, device=x.device).unsqueeze(0).expand(batch_size, max_time)
        non_blank_ids_mask = torch.logical_and(predictions_labels != self.blank_id, time_steps < out_len.unsqueeze(1))
        predictions_labels = predictions_labels.cpu()
        out_len = out_len.cpu()

        hypotheses = []

        for i in range(batch_size):
            hypothesis = rnnt_utils.Hypothesis(score=0.0, y_sequence=[], dec_state=None, timestamp=[], last_token=None)
            hypothesis.y_sequence = predictions_labels[i, : out_len[i]].tolist()
            hypothesis.score = -1.0

            if self.preserve_alignments:
                raise ValueError(
                    "Requested for alignments, but predictions provided were labels, not log probabilities."
                )
            if self.compute_timestamps:
                # TOOD: Could do this in a vectorized manner... Would
                # prefer to have nonzero_static, though, for sanity.
                # Or do a prefix sum on out_len
                hypothesis.timestamp = torch.nonzero(non_blank_ids_mask[i], as_tuple=False)[:, 0].cpu().tolist()
            if self.preserve_frame_confidence:
                raise ValueError(
                    "Requested for per-frame confidence, but predictions provided were labels, not log probabilities."
                )

            hypotheses.append(hypothesis)

        return hypotheses

    @torch.no_grad()
    def _greedy_decode_logprobs_batched_lm_loop(self, logits: torch.Tensor, out_len: torch.Tensor):
        batch_size = logits.shape[0]
        max_time = logits.shape[1]
        device = logits.device
        float_dtype = logits.dtype
        
        if self.ngram_lm_batch is not None:
            batch_lm_states = self.ngram_lm_batch.get_init_states(batch_size=batch_size, bos=True)
        if self.wb_batch is not None:
            batch_wb_states = self.wb_batch.get_init_states(batch_size=batch_size, bos=False)
            
        predictions_labels = torch.zeros([batch_size, max_time], device=device, dtype=torch.long)
        predictions_logprobs = torch.zeros([batch_size, max_time], device=device, dtype=float_dtype)
        batch_indices = torch.arange(batch_size, device=device, dtype=torch.long)
        last_labels = torch.full([batch_size], fill_value=self.blank_id, device=device, dtype=torch.long)

        for i in range(max_time):
            labels = torch.argmax(logits[:, i], dim=-1)
            log_probs_w_lm = logits[:, i]  # .clone()
            
            if self.ngram_lm_batch is not None:
                lm_scores, batch_lm_states_candidates = self.ngram_lm_batch.advance(states=batch_lm_states)
                lm_scores = lm_scores.to(dtype=float_dtype)
                log_probs_w_lm[:, :-1] += self.ngram_lm_alpha * lm_scores
            if self.wb_batch is not None:
                wb_scores, batch_wb_states_candidates = self.wb_batch.advance(states=batch_wb_states)
                wb_scores = wb_scores.to(dtype=float_dtype)
                log_probs_w_lm[:, :-1] += self.btree_alpha * wb_scores

            # log_probs_w_lm[:, -1] = NEG_INF - no need, argmax without last label
            # use scatter instead of
            # log_probs_w_lm[batch_indices, last_labels] = NEG_INF
            log_probs_w_lm.scatter_(dim=1, index=last_labels.unsqueeze(-1), value=NEG_INF)
            labels_w_lm = log_probs_w_lm[:, :-1].argmax(dim=-1)

            blank_or_repeated = (labels == self.blank_id) | (labels == last_labels)
            torch.where(blank_or_repeated, labels, labels_w_lm, out=labels)
            if self.ngram_lm_batch is not None:
                torch.where(
                    blank_or_repeated,
                    batch_lm_states,
                    batch_lm_states_candidates[batch_indices, labels * ~blank_or_repeated],
                    out=batch_lm_states,
                )
            if self.wb_batch is not None:
                torch.where(
                    blank_or_repeated,
                    batch_wb_states,
                    batch_wb_states_candidates[batch_indices, labels * ~blank_or_repeated],
                    out=batch_wb_states,
                )
            predictions_labels[:, i] = labels
            # TODO: logprobs
            last_labels = labels
        return predictions_labels, predictions_logprobs

    @torch.no_grad()
    def _before_loop(self):
        self._cuda_graphs_state.batch_lm_states.copy_(
            self.ngram_lm_batch.get_init_states(batch_size=self._cuda_graphs_state.batch_size, bos=True)
        )
        self._cuda_graphs_state.predictions_labels.zero_()
        self._cuda_graphs_state.predictions_logprobs.zero_()
        self._cuda_graphs_state.last_labels.fill_(self.blank_id)

        self._cuda_graphs_state.frame_i.fill_(0)
        self._cuda_graphs_state.decoding_active.copy_((self._cuda_graphs_state.logits_len > 0).any())

    @torch.no_grad()
    def _inner_loop_code(self):
        lm_scores, batch_lm_states_candidates = self.ngram_lm_batch.advance(
            states=self._cuda_graphs_state.batch_lm_states
        )
        lm_scores = lm_scores.to(dtype=self._cuda_graphs_state.float_dtype)
        cur_logits = self._cuda_graphs_state.logits[:, self._cuda_graphs_state.frame_i.unsqueeze(0)].squeeze(1)
        # cur_logits = torch.index_select(self._cuda_graphs_state.logits, dim=1, index=self._cuda_graphs_state.frame_i).squeeze(1)
        labels = torch.argmax(cur_logits, dim=-1)
        log_probs_w_lm = cur_logits.clone()
        log_probs_w_lm[:, :-1] += self.ngram_lm_alpha * lm_scores
        # log_probs_w_lm[:, -1] = NEG_INF - no need, argmax without last label
        # use scatter instead of
        # log_probs_w_lm[batch_indices, last_labels] = NEG_INF
        log_probs_w_lm.scatter_(dim=1, index=self._cuda_graphs_state.last_labels.unsqueeze(-1), value=NEG_INF)
        labels_w_lm = log_probs_w_lm[:, :-1].argmax(dim=-1)

        blank_or_repeated = (labels == self.blank_id) | (labels == self._cuda_graphs_state.last_labels)
        torch.where(blank_or_repeated, labels, labels_w_lm, out=labels)
        torch.where(
            blank_or_repeated,
            self._cuda_graphs_state.batch_lm_states,
            batch_lm_states_candidates[self._cuda_graphs_state.batch_indices, labels * ~blank_or_repeated],
            out=self._cuda_graphs_state.batch_lm_states,
        )
        self._cuda_graphs_state.predictions_labels[:, self._cuda_graphs_state.frame_i.unsqueeze(0)] = labels.unsqueeze(
            -1
        )
        # # TODO: logprobs
        self._cuda_graphs_state.last_labels.copy_(labels)
        self._cuda_graphs_state.frame_i += 1
        self._cuda_graphs_state.decoding_active.copy_(
            (self._cuda_graphs_state.logits_len > self._cuda_graphs_state.frame_i).any()
        )

    @classmethod
    def _create_while_loop_kernel(cls):
        """
        Creates a kernel that evaluates whether to enter the outer loop body (not all hypotheses are decoded).
        Condition: while(active_mask_any).
        """
        kernel_string = r"""\
            typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;

            extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);

            extern "C" __global__
            void ctc_loop_conditional(cudaGraphConditionalHandle handle, const bool *decoding_active)
            {
             cudaGraphSetConditional(handle, *decoding_active);
            }
            """
        return run_nvrtc(kernel_string, b"ctc_loop_conditional", b"while_conditional_ctc.cu")

    def _graph_reinitialize(self, logits, logits_len):
        batch_size, max_time, encoder_dim = logits.shape

        self._cuda_graphs_state = CTCDecoderCudaGraphsState(
            batch_size=batch_size,
            max_time=max(max_time, 375),
            encoder_dim=encoder_dim,
            device=logits.device,
            float_dtype=logits.dtype,
        )
        use_full_cuda_graphs = True
        if use_full_cuda_graphs:
            stream_for_graph = torch.cuda.Stream(self._cuda_graphs_state.device)
            stream_for_graph.wait_stream(torch.cuda.default_stream(self._cuda_graphs_state.device))
            self._cuda_graphs_state.full_graph = torch.cuda.CUDAGraph()
            with (
                torch.cuda.stream(stream_for_graph),
                torch.inference_mode(),
                torch.cuda.graph(
                    self._cuda_graphs_state.full_graph, stream=stream_for_graph, capture_error_mode="thread_local"
                ),
            ):
                self._before_loop()

                capture_status, _, graph, _, _ = cu_call(
                    cudart.cudaStreamGetCaptureInfo(
                        torch.cuda.current_stream(device=self._cuda_graphs_state.device).cuda_stream
                    )
                )
                assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

                # capture: while decoding_active:
                (loop_conditional_handle,) = cu_call(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
                loop_kernel = self._create_while_loop_kernel()
                decoding_active_ptr = np.array([self._cuda_graphs_state.decoding_active.data_ptr()], dtype=np.uint64)
                loop_args = np.array(
                    [loop_conditional_handle.getPtr(), decoding_active_ptr.ctypes.data],
                    dtype=np.uint64,
                )
                # loop while there are active utterances
                with with_conditional_node(
                    loop_kernel,
                    loop_args,
                    loop_conditional_handle,
                    device=self._cuda_graphs_state.device,
                ):
                    self._inner_loop_code()
        else:
            stream_for_graph = torch.cuda.Stream(self._cuda_graphs_state.device)
            stream_for_graph.wait_stream(torch.cuda.default_stream(self._cuda_graphs_state.device))
            self._cuda_graphs_state.before_loop_graph = torch.cuda.CUDAGraph()
            self._cuda_graphs_state.inner_loop_graph = torch.cuda.CUDAGraph()
            with (
                torch.cuda.stream(stream_for_graph),
                torch.inference_mode(),
                torch.cuda.graph(
                    self._cuda_graphs_state.before_loop_graph,
                    stream=stream_for_graph,
                    capture_error_mode="thread_local",
                ),
            ):
                self._before_loop()

            with (
                torch.cuda.stream(stream_for_graph),
                torch.inference_mode(),
                torch.cuda.graph(
                    self._cuda_graphs_state.inner_loop_graph,
                    stream=stream_for_graph,
                    capture_error_mode="thread_local",
                ),
            ):
                self._inner_loop_code()

    def _greedy_decode_logprobs_batched_lm_loop_cuda_graphs(self, logits: torch.Tensor, out_len: torch.Tensor):
        current_batch_size = logits.shape[0]
        current_max_time = logits.shape[1]

        if torch.is_autocast_enabled():
            logits = logits.to(torch.get_autocast_gpu_dtype())

        # init or reinit graph
        if self._cuda_graphs_state is None or self._cuda_graphs_state.need_reinit(logits):
            self._graph_reinitialize(logits=logits, logits_len=out_len)

        # copy (projected) encoder output and lenghts
        self._cuda_graphs_state.logits[:current_batch_size, :current_max_time, ...].copy_(logits)
        self._cuda_graphs_state.logits_len[: logits.shape[0]].copy_(out_len)
        # set length to zero for elements outside the current batch
        self._cuda_graphs_state.logits_len[current_batch_size:].fill_(0)
        # if self.cuda_graphs_mode is self.CudaGraphsMode.FULL_GRAPH:
        # self._cuda_graphs_state.full_graph.replay()

        # self._before_loop()
        # while self._cuda_graphs_state.decoding_active:
        #     self._inner_loop_code()

        self._cuda_graphs_state.full_graph.replay()

        # self._cuda_graphs_state.before_loop_graph.replay()
        # for _ in range(current_max_time):
        #     self._cuda_graphs_state.inner_loop_graph.replay()

        return (
            self._cuda_graphs_state.predictions_labels[:current_batch_size, :current_max_time].clone(),
            self._cuda_graphs_state.predictions_logprobs[:current_batch_size, :current_max_time].clone(),
        )

    @torch.no_grad()
    def _greedy_decode_logprobs_batched_lm(self, x: torch.Tensor, out_len: torch.Tensor):
        # x: [B, T, D]
        # out_len: [B]

        batch_size = x.shape[0]
        max_time = x.shape[1]

        log_probs = x

        predictions_labels, predictions_logprobs = self._greedy_decode_logprobs_batched_lm_loop(
            logits=x, out_len=out_len
        )
        # predictions_labels, predictions_logprobs = self._greedy_decode_logprobs_batched_lm_loop_cuda_graphs(
        #     logits=x, out_len=out_len
        # )

        # In CTC greedy decoding, each output maximum likelihood token
        # is calculated independent of the other tokens.
        # predictions_logprobs, predictions_labels = predictions.max(dim=-1)

        # Since predictions_logprobs is a padded matrix in the time
        # dimension, we consider invalid timesteps to be "blank".
        time_steps = torch.arange(max_time, device=x.device).unsqueeze(0).expand(batch_size, max_time)
        non_blank_ids_mask = torch.logical_and(predictions_labels != self.blank_id, time_steps < out_len.unsqueeze(1))
        # Sum the non-blank labels to compute the score of the
        # transcription. This follows from Eq. (3) of "Connectionist
        # Temporal Classification: Labelling Unsegmented Sequence Data
        # with Recurrent Neural Networks".
        scores = torch.where(non_blank_ids_mask, predictions_logprobs, 0.0).sum(axis=1)

        scores = scores.cpu()
        predictions_labels = predictions_labels.cpu()
        out_len = out_len.cpu()

        predictions = log_probs
        if self.preserve_alignments or self.preserve_frame_confidence:
            predictions = predictions.cpu()

        hypotheses = []

        # This mimics the for loop in GreedyCTCInfer::forward.
        for i in range(batch_size):
            hypothesis = rnnt_utils.Hypothesis(score=0.0, y_sequence=[], dec_state=None, timestamp=[], last_token=None)
            hypothesis.score = scores[i]

            prediction_labels_no_padding = predictions_labels[i, : out_len[i]].tolist()

            assert predictions_labels.dtype == torch.int64
            hypothesis.y_sequence = prediction_labels_no_padding

            if self.preserve_alignments:
                hypothesis.alignments = (
                    predictions[i, : out_len[i], :].clone(),
                    predictions_labels[i, : out_len[i]].clone(),
                )
            if self.compute_timestamps:
                # TOOD: Could do this in a vectorized manner... Would
                # prefer to have nonzero_static, though, for sanity.
                # Or do a prefix sum on out_len
                hypothesis.timestep = torch.nonzero(non_blank_ids_mask[i], as_tuple=False)[:, 0].cpu().tolist()
            if self.preserve_frame_confidence:
                hypothesis.frame_confidence = self._get_confidence(predictions[i, : out_len[i], :])

            hypotheses.append(hypothesis)

        return hypotheses

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


@dataclass
class GreedyCTCInferConfig:
    preserve_alignments: bool = False
    compute_timestamps: bool = False
    preserve_frame_confidence: bool = False
    confidence_method_cfg: Optional[ConfidenceMethodConfig] = field(default_factory=lambda: ConfidenceMethodConfig())

    ngram_lm_model: Optional[str] = None
    ngram_lm_alpha: float = 0.0

    btree_model: Optional[str] = None
    btree_alpha: float = 0.0
    
    def __post_init__(self):
        # OmegaConf.structured ensures that post_init check is always executed
        self.confidence_method_cfg = OmegaConf.structured(
            self.confidence_method_cfg
            if isinstance(self.confidence_method_cfg, ConfidenceMethodConfig)
            else ConfidenceMethodConfig(**self.confidence_method_cfg)
        )
