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
from typing import Any, Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F

from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.utils.ctc_batched_beam_utils import BatchedBeamHypsCTC
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodMixin
from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs
from nemo.core.utils.cuda_python_utils import (
    check_cuda_python_cuda_graphs_conditional_nodes_supported,
    cu_call,
    run_nvrtc,
    with_conditional_node,
)
from nemo.utils import logging
from nemo.utils.enum import PrettyStrEnum
from nemo.collections.asr.parts.utils.ctc_batched_beam_utils import BlankLMScoreMode

try:
    from cuda import cudart

    HAVE_CUDA_PYTHON = True
except ImportError:
    HAVE_CUDA_PYTHON = False


NON_EXISTENT_LABEL_VALUE=-1
INACTIVE_SCORE=float("-inf")


class BacthedBeamCTCState:
    """
    State for batched ALSD algorithm for RNN-T models. Used only with CUDA graphs.
    In initialization phase it is possible to assign values (tensors) to the state.
    For algorithm code the storage should be reused (prefer copy data instead of assigning tensors).
    """
    
    max_time: int  # maximum length of internal storage for time dimension
    batch_size: int  # (maximum) length of internal storage for batch dimension
    device: torch.device  # device to store preallocated tensors
    beam_size: int  # (maximum) length of internal storage for beam dimension
    blank_index: int  # the index of the blank token

    decoder_outputs: torch.Tensor
    decoder_output_lengths: torch.Tensor
    last_timesteps: torch.Tensor

    curr_length: torch.Tensor
    
    log_probs: torch.Tensor
    next_labels: torch.Tensor  # storage for next labels
    next_scores: torch.Tensor  # storage for next scores
    next_indices: torch.Tensor  # storage for next scores
    total_scores: torch.Tensor
    lm_scores: torch.Tensor

    batch_indices: torch.Tensor  # indices of elements in batch (constant, range [0, batch_size-1])
    beam_indices: torch.Tensor  # indices of elements in batch (constant, range [0, beam_size-1])
    batch_indices1d: torch.Tensor

    blank_mask: torch.Tensor
    active_mask: torch.Tensor  # mask for active hypotheses (the decoding is finished for the utterance if it is False)
    repeated_mask: torch.Tensor
    active_mask_any: torch.Tensor  # 0-dim bool tensor, condition for outer loop ('any element is still active')
    vocab: torch.Tensor
    batch_labels: torch.Tensor
    zeros_column: torch.Tensor

    batched_hyps: BatchedBeamHypsCTC  # batched hypotheses - decoding result

    # LM-related fields
    ngram_lm_batch: Optional[NGramGPULanguageModel] = None  # N-gram LM for hypotheses
    lm_scores: Optional[torch.Tensor] = None  # LM scores for hypotheses
    batch_lm_states: Optional[torch.Tensor] = None  # LM states for hypotheses
    batch_lm_states_candidates: Optional[torch.Tensor] = None  # LM states for hypotheses candidates
    batch_lm_states_prev: Optional[torch.Tensor] = None  # previous LM states for hypotheses
    init_lm_scores: Optional[torch.Tensor] = None  # initial LM scores for hypotheses
    init_batch_lm_states: Optional[torch.Tensor] = None  # initial LM states for hypotheses
    init_batch_lm_states_candidates: Optional[torch.Tensor] = None  # initial LM states for hypotheses candidates
    
    log10e: torch.Tensor

    def __init__(
        self,
        batch_size: int,
        beam_size: int,
        max_time: int,
        vocab_size: int,
        device: torch.device,
        float_dtype: torch.dtype,
        blank_index: int,
    ):
        """
        Args:
            batch_size: batch size for encoder output storage
            beam_size: beam size for decoder output storage
            max_time: maximum time for encoder output storage
            encoder_dim: last dimension for encoder output storage (projected encoder output)
            device: device to store tensors
            float_dtype: default float dtype for tensors (should match projected encoder output)
            blank_index: index of the blank symbol
        """

        self.device = device
        self.float_dtype = float_dtype
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.max_time = max_time
        self.blank_index = blank_index
        self.vocab_size = vocab_size
        
        self.MINUS_ONE_TENSOR=torch.tensor(-1, device=self.device, dtype=torch.long)

        self.NON_EXISTENT_LABEL = torch.tensor(NON_EXISTENT_LABEL_VALUE, device=self.device, dtype=torch.long)
        self.BLANK_TENSOR = torch.tensor(self.blank_index, device=self.device, dtype=torch.long)
        self.INACTIVE_SCORE = torch.tensor(INACTIVE_SCORE, device=self.device, dtype=float_dtype)

        self.decoder_outputs = torch.zeros(
            (self.batch_size, self.max_time, self.vocab_size),
            dtype=float_dtype,
            device=self.device,
        )
        self.decoder_output_lengths = torch.zeros(
            (self.batch_size, self.beam_size),
            dtype=torch.long,
            device=self.device
        )
        self.last_timesteps = torch.zeros(
            (self.batch_size, self.beam_size),
            dtype=torch.long,
            device=self.device
        )

        self.next_labels = torch.zeros([self.batch_size, self.beam_size], dtype=torch.long, device=self.device)
        self.next_scores = torch.zeros([self.batch_size, self.beam_size], dtype=float_dtype, device=self.device)
        self.next_indices = torch.zeros([self.batch_size, self.beam_size], dtype=torch.long, device=self.device)
        self.lm_scores = torch.zeros([self.batch_size, self.beam_size, self.vocab_size], dtype=self.float_dtype, device=self.device)
        
        self.total_scores = torch.full(
            [self.batch_size, self.beam_size],
            fill_value=self.INACTIVE_SCORE,
            device=self.device,
            dtype=float_dtype
        )
        self.log_probs = torch.zeros([self.batch_size, self.beam_size, self.vocab_size],  dtype=float_dtype, device=self.device)

        # indices of elements in batch and beam (constant)
        self.batch_indices = (
            torch.arange(batch_size, dtype=torch.long, device=device)[:, None]
            .expand(batch_size, self.beam_size)
            .clone()
        )  # size: batch_size x beam_size
        self.beam_indices = (
            torch.arange(self.beam_size, dtype=torch.long, device=self.device)[None, :, None]
            .expand(self.batch_size, -1, self.beam_size)
            .clone()
        )  # size: batch_size x beam_size x beam_size

        self.active_mask = torch.zeros_like(self.batch_indices, dtype=torch.bool)
        self.active_mask_any = torch.tensor(True, device=self.device, dtype=torch.bool)
        self.vocab=torch.arange(self.vocab_size, device=self.device, dtype=torch.long)
        self.batch_indices1d=torch.arange(self.batch_size, device=self.device, dtype=torch.long)
        self.batch_labels = self.vocab[None, None, :].expand(batch_size, self.beam_size, -1)
        self.non_blank_mask = self.batch_labels != self.blank_index
        self.zeros_column = torch.zeros([self.batch_size, self.beam_size, 1], device=self.device, dtype=self.float_dtype)
        self.false_mask = torch.zeros([self.batch_size, self.beam_size, self.vocab_size], dtype=torch.bool, device=self.device)
        self.curr_length = torch.tensor(0, device=self.device, dtype=torch.long)
        
        self.repeated_mask = torch.zeros([self.batch_size, self.beam_size, self.vocab_size], device=self.device, dtype=torch.bool)
        self.repeated_or_blank_mask = torch.zeros((self.batch_size, self.beam_size, self.vocab_size), device=self.device, dtype=torch.bool)
        self.blank_mask = torch.eq(self.vocab, self.blank_index)

        self.batched_hyps = BatchedBeamHypsCTC(
            batch_size=batch_size,
            beam_size=self.beam_size,
            blank_index=self.blank_index,
            init_length=max_time + 1,
            device=device,
            float_dtype=float_dtype,
        )
        
        self.log10e = torch.log10(torch.tensor(torch.e, device=self.device, dtype=self.float_dtype))

    def need_reinit(self, encoder_output_projected: torch.Tensor) -> bool:
        """Check if need to reinit state: larger batch_size/max_time, or new device"""
        return (
            self.batch_size < encoder_output_projected.shape[0]
            or self.max_time < encoder_output_projected.shape[1]
            or self.device.index != encoder_output_projected.device.index
        )


@dataclass
class SeparateGraphsBatchedBeamCTC:
    """Class to store Cuda graphs for decoding when separate graphs are used"""

    _before_process_batch: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)
    _process_batch: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)
    _after_process_batch: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)


class BatchedBeamCTCComputer(WithOptionalCudaGraphs, ConfidenceMethodMixin):
    """
    Batched Alignment-Length Synchronous Decoding implementation. Callable.
    Based on https://ieeexplore.ieee.org/document/9053040 with the following modficiations:
        - does not support prediction network caching
        - does not employ transcript length estimation, instead, limits the number of expansions for every frame.
    """

    INITIAL_MAX_TIME = 375  # initial max time, used to init state for Cuda graphs
    CUDA_PROGRAM_NAME = b"while_malsd_batch_conditional_rnnt.cu"

    class CudaGraphsMode(PrettyStrEnum):
        FULL_GRAPH = "full_graph"  # Cuda graphs with conditional nodes, fastest implementation
        NO_WHILE_LOOPS = "no_while_loops"  # Decoding with PyTorch while loops + partial Cuda graphs
        NO_GRAPHS = "no_graphs"  # decoding without graphs, stateful implementation, only for testing purposes

    separate_graphs: Optional[SeparateGraphsBatchedBeamCTC]
    full_graph: Optional[torch.cuda.CUDAGraph]
    cuda_graphs_mode: Optional[CudaGraphsMode]
    state: Optional[BacthedBeamCTCState]
    ngram_lm_batch: Optional[NGramGPULanguageModel]

    def __init__(
        self,
        blank_index: int,
        beam_size: int,
        return_best_hypothesis: bool = True,
        preserve_alignments=False,
        compute_timestamps: bool = False,
        beam_alpha: float = 1.0,
        beam_beta: float = 0.0,
        beam_threshold: float = 20.0,
        beam_size_token: Optional[int] = 16,
        blank_lm_score_mode: str = "no_score",
        kenlm_path: str = None,
        allow_cuda_graphs: bool = True
    ):
        """
        Init method.
        Args:
            decoder: Prediction network from RNN-T
            joint: Joint module from RNN-T
            blank_index: index of blank symbol
            max_symbols_per_step: max symbols to emit on each step (to avoid infinite looping)
            preserve_alignments: if alignments are needed
            preserve_frame_confidence: if frame confidence is needed
            confidence_method_cfg: config for the confidence
            ngram_lm_model: path to the NGPU-LM n-gram LM model: .arpa or .nemo formats
            ngram_lm_alpha: weight for the n-gram LM scores
            blank_lm_score_mode: mode for scoring blank symbol with LM
            pruning_mode: mode for pruning hypotheses with LM
            score_norm: whether to normalize scores before best hypothesis extraction
            allow_cuda_graphs: whether to allow CUDA graphs
            return_best_hypothesis: whether to return the best hypothesis or N-best hypotheses
        """

        super().__init__()
        self._blank_index = blank_index

        self.beam_size = beam_size
        self.preserve_alignments = preserve_alignments
        self.compute_timestamps = compute_timestamps
        self.allow_cuda_graphs = allow_cuda_graphs
        self.return_best_hypothesis = return_best_hypothesis
        
        self.beam_alpha = beam_alpha
        self.beam_beta = beam_beta
        self.beam_threshold = beam_threshold
        self.beam_size_token = beam_size_token

        assert not self.preserve_alignments, "Preserve aligments is not supported"
        assert not self.compute_timestamps, "Compute timestamps is not supported"

        self.state = None
        self.full_graph = None
        self.separate_graphs = None

        # prprprprpr
        self.cuda_graphs_mode = None
        self.maybe_enable_cuda_graphs()
        # self.cuda_graphs_mode = self.CudaGraphsMode.NO_GRAPHS

        self.ngram_lm_batch = None
        if kenlm_path is not None:
            assert self._blank_index == 1024
            self.ngram_lm_batch = NGramGPULanguageModel.from_file(lm_path=kenlm_path, vocab_size=self._blank_index)

    def force_cuda_graphs_mode(self, mode: Optional[Union[str, CudaGraphsMode]]):
        """
        Method to set graphs mode. Use only for testing purposes.
        For debugging the algorithm use "no_graphs" mode, since it is impossible to debug CUDA graphs directly.
        """
        self.cuda_graphs_mode = self.CudaGraphsMode(mode) if mode is not None else None
        self.state = None

    def maybe_enable_cuda_graphs(self):
        """Enable CUDA graphs if conditions met"""
        if self.cuda_graphs_mode is not None:
            # CUDA graphs are already enabled
            return

        if not self.allow_cuda_graphs:
            self.cuda_graphs_mode = None
        else:
            # cuda graphs are allowed
            # check while loops
            try:
                check_cuda_python_cuda_graphs_conditional_nodes_supported()
                self.cuda_graphs_mode = self.CudaGraphsMode.FULL_GRAPH
            except (ImportError, ModuleNotFoundError, EnvironmentError) as e:
                logging.warning(
                    "No conditional node support for Cuda.\n"
                    "Cuda graphs with while loops are disabled, decoding speed will be slower\n"
                    f"Reason: {e}"
                )
                self.cuda_graphs_mode = self.CudaGraphsMode.NO_GRAPHS
        self.reset_cuda_graphs_state()

    def disable_cuda_graphs(self):
        """Disable CUDA graphs, can be used to disable graphs temporary, e.g., in training process"""
        if self.cuda_graphs_mode is None:
            # nothing to disable
            return
        self.cuda_graphs_mode = None
        self.reset_cuda_graphs_state()

    def reset_cuda_graphs_state(self):
        """Reset state to release memory (for CUDA graphs implementations)"""
        self.state = None
        self.full_graph = None
        self.separate_graphs = None

    @torch.no_grad()
    def batched_beam_search_torch(
        self, decoder_outputs: torch.Tensor, decoder_output_lengths: torch.Tensor
    ) -> List[Union[rnnt_utils.Hypothesis, rnnt_utils.NBestHypotheses]]:
        """
        Open Seq2Seq Beam Search Algorithm (DeepSpeed)

        Args:
            x: Tensor of shape [B, T, V+1], where B is the batch size, T is the maximum sequence length,
                and V is the vocabulary size. The tensor contains log-probabilities.
            out_len: Tensor of shape [B], contains lengths of each sequence in the batch.

        Returns:
            A list of NBestHypotheses objects, one for each sequence in the batch.
        """              
        batch_size, max_time, vocab_size = decoder_outputs.shape

        zeros_column = torch.zeros([batch_size, self.beam_size, 1], device=decoder_outputs.device, dtype=torch.long)
        vocab = torch.arange(vocab_size, device=decoder_outputs.device, dtype=torch.long)
        batch_labels = vocab[None, None, :].expand(batch_size, self.beam_size, -1)
        
        batched_beam_hyps = BatchedBeamHypsCTC(
            batch_size=batch_size,
            beam_size=self.beam_size,
            blank_index=self._blank_index,
            init_length=max_time + 1,
            device=decoder_outputs.device,
            float_dtype=decoder_outputs.dtype,
        )
        
        if self.ngram_lm_batch is not None:
            self.ngram_lm_batch.to(decoder_outputs.device)
            batch_lm_states = self.ngram_lm_batch.get_init_states(batch_size=batch_size * self.beam_size, bos=True)
        
        # step=0
        for t in range(max_time):
            active_mask = decoder_output_lengths.unsqueeze(1) > t
            log_probs = decoder_outputs[:, t, :].unsqueeze(1) * np.log10(np.e)
            
            # if self.beam_size_token is not None:
            #     _, topk_idx = torch.topk(log_probs, k=self.beam_size_token, largest=True, sorted=True)
            #     mask = torch.zeros_like(log_probs, dtype=torch.bool).scatter_(dim=2, index=topk_idx, value=True)
            #     log_probs.masked_fill_(~mask, float('inf'))
            
            if self.ngram_lm_batch is not None:
                lm_scores, batch_lm_states_candidates = self.ngram_lm_batch.advance(states=batch_lm_states)
                lm_scores = lm_scores.to(dtype=decoder_outputs.dtype).view(batch_size, self.beam_size, -1) * self.beam_alpha * np.log10(np.e)
            
                log_probs = log_probs + torch.cat((lm_scores, zeros_column), dim=-1)
                            
            repeated_mask = batched_beam_hyps.last_label[:, :, None] == vocab[None, None, :]
            blank_mask = (vocab == self._blank_index)[None, None, :]
            
            total_scores = batched_beam_hyps.scores[:, :, None] + log_probs
            total_scores = torch.where(blank_mask | repeated_mask, total_scores, total_scores + self.beam_beta)
            
            hyps_scores, hyps_candidates_indices = torch.topk(total_scores.view(batch_size, -1), k=self.beam_size, largest=True, sorted=True)
            hyps_indices = hyps_candidates_indices // vocab_size # torch.gather(expansion_indices.reshape(batch_size, -1), dim=-1, index=hyps_candidates_indices)
            next_labels = torch.gather(batch_labels, dim=-1, index=(hyps_candidates_indices % vocab_size).unsqueeze(-1)).squeeze(2)
            hyps_scores.view(batch_size, -1)[
                hyps_scores.view(batch_size, -1) <= 
                hyps_scores.view(batch_size, -1).max(dim=-1, keepdim=True).values - 
                self.beam_threshold
            ] = float('-inf')
            
            repeating_mask = next_labels == torch.gather(batched_beam_hyps.last_label, dim=-1, index=hyps_indices)
            blank_mask = next_labels == self._blank_index
            
            preserve_state_mask = repeating_mask | blank_mask | ~ active_mask
            next_labels_masked = torch.where(blank_mask, 0, next_labels)
                
            if self.ngram_lm_batch is not None:
                # batch_lm_states: [(BxBeam)]
                # batch_lm_states_candidates: [(BxBeam) x V (without blank)]
                batch_lm_states_candidates = torch.gather(
                    batch_lm_states_candidates.view(batch_size, self.beam_size, -1),
                    dim=1,
                    index=hyps_indices[:, :, None].expand(
                        batch_size, self.beam_size, batch_lm_states_candidates.shape[-1]
                    ),
                )
                batch_lm_states_prev = torch.gather(
                    batch_lm_states.view(batch_size, self.beam_size), dim=1, index=hyps_indices
                )

                batch_lm_states = torch.gather(
                    batch_lm_states_candidates, dim=-1, index=next_labels_masked.unsqueeze(-1)
                ).squeeze(-1)
                batch_lm_states = torch.where(preserve_state_mask, batch_lm_states_prev, batch_lm_states).view(-1)
        
            next_labels = torch.where(active_mask, next_labels, -1)
            batched_beam_hyps.add_results_(hyps_indices, next_labels, hyps_scores)
            
            batched_beam_hyps.self_recombine_hyps_()
            
            # print("Step: ", step)
            # print("scores: ", batched_beam_hyps.scores)
            # print("labels: ", batched_beam_hyps.transcript_wb[..., step])
            # print("ptrs: ", batched_beam_hyps.transcript_wb_prev_ptr[..., step])
            # step+=1

        if self.ngram_lm_batch is not None:
            batched_beam_hyps.scores += self.ngram_lm_batch.get_final(batch_lm_states).view(batch_size, self.beam_size) * np.log10(np.e) * self.beam_alpha

        # for hyp in batched_beam_hyps.to_hyps_list():
        #     print("score", hyp.score)
        #     print("y_sequence", hyp.y_sequence)
            
        nbest_hypotheses = []
        for decoder_outputs in batched_beam_hyps.to_hyps_list():
            # Wrap the result in NBestHypothesis.
            hypotheses = rnnt_utils.NBestHypotheses([decoder_outputs])
            nbest_hypotheses.append(hypotheses)
            
        return nbest_hypotheses

    def batched_beam_search_cuda_graphs(
        self,
        decoder_outputs: torch.Tensor,
        decoder_output_lengths: torch.Tensor,
    ) -> Tuple[rnnt_utils.BatchedHyps, Optional[rnnt_utils.BatchedAlignments], Any]:
        assert self.cuda_graphs_mode is not None

        current_batch_size = decoder_outputs.shape[0]
        current_max_time = decoder_outputs.shape[1]

        if torch.is_autocast_enabled():
            decoder_outputs = decoder_outputs.to(torch.get_autocast_gpu_dtype())

        # init or reinit graph
        if self.state is None or self.state.need_reinit(decoder_outputs):
            self._graph_reinitialize(decoder_outputs, decoder_output_lengths)

        # set length to zero for elements outside the current batch
        self.state.decoder_output_lengths.fill_(0)
        # copy (projected) encoder output and lenghts
        self.state.decoder_outputs[:current_batch_size, :current_max_time, ...].copy_(decoder_outputs)
        self.state.decoder_output_lengths[:current_batch_size].copy_(decoder_output_lengths.unsqueeze(-1))
        if self.cuda_graphs_mode is self.CudaGraphsMode.FULL_GRAPH:
            self.full_graph.replay()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_WHILE_LOOPS:
            self.separate_graphs._before_process_batch.replay()
            while self.state.active_mask_any.item():
                self.separate_graphs._process_batch.replay()
            self.separate_graphs._after_process_batch.replay()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_GRAPHS:
            # this mode is only for testing purposes
            # manual loop instead of using graphs
            # step=0
            self._before_process_batch()
            while self.state.active_mask_any.item():
                self._process_batch()
            
                # print("Step: ", step)
                # print("scores: ", self.state.batched_hyps.scores)
                # print("labels: ", self.state.batched_hyps.transcript_wb[..., step])
                # print("ptrs: ", self.state.batched_hyps.transcript_wb_prev_ptr[..., step])
                # step+=1
            
            self._after_process_batch()
            
            # for hyp in self.state.batched_hyps.to_hyps_list():
            #     print("score", hyp.score)
            #     print("y_sequence", hyp.y_sequence)
        else:
            raise NotImplementedError(f"Unknown graph mode: {self.cuda_graphs_mode}")

        nbest_hypotheses = []
        for decoder_outputs in self.state.batched_hyps.to_hyps_list():
            # Wrap the result in NBestHypothesis.
            hypotheses = rnnt_utils.NBestHypotheses([decoder_outputs])
            nbest_hypotheses.append(hypotheses)
            
        return nbest_hypotheses[:current_batch_size]

    @classmethod
    def _create_process_batch_kernel(cls):
        """
        Creates a kernel that evaluates whether to enter the outer loop body (not all hypotheses are decoded).
        Condition: while(active_mask_any).
        """
        kernel_string = r"""\
        typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;
    
        extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);
    
        extern "C" __global__
        void loop_conditional(cudaGraphConditionalHandle handle, const bool *active_mask_any)
        {
         cudaGraphSetConditional(handle, *active_mask_any);
        }
        """
        return run_nvrtc(kernel_string, b"loop_conditional", cls.CUDA_PROGRAM_NAME)

    def _graph_reinitialize(
        self,
        decoder_outputs: torch.Tensor,
        decoder_output_lengths: torch.Tensor,
    ):
        """
        Reinitializes the graph state for the MALSD computation.
        This method sets up the internal state required for the decoding process, including initializing
        decoder outputs, decoder states, and optional n-gram language model states. It also handles CUDA
        graph compilation based on the specified mode.
        Args:
            encoder_output_projected (torch.Tensor): The projected encoder output tensor of shape
                (batch_size, max_time, encoder_dim).
            encoder_output_length (torch.Tensor): The lengths of the encoder outputs for each batch.
        Raises:
            NotImplementedError: If an unsupported CUDA graph mode is specified.
        """

        batch_size, max_time, vocab_size = decoder_outputs.shape

        self.state = BacthedBeamCTCState(
            batch_size=batch_size,
            beam_size=self.beam_size,
            max_time=max(max_time, self.INITIAL_MAX_TIME),
            vocab_size=vocab_size,
            device=decoder_outputs.device,
            float_dtype=decoder_outputs.dtype,
            blank_index=self._blank_index,
        )

        if self.ngram_lm_batch is not None:
            device = decoder_outputs.device

            self.ngram_lm_batch.to(device)

            self.state.init_batch_lm_states = self.ngram_lm_batch.get_init_states(
                batch_size=self.state.batch_size * self.beam_size, bos=True
            ).view(self.state.batch_size, self.beam_size)
            init_lm_scores, init_batch_lm_states_candidates = self.ngram_lm_batch.advance(
                states=self.state.init_batch_lm_states.view(-1)
            )  # vocab_size_no_blank
            self.state.init_lm_scores = (
                init_lm_scores.to(dtype=self.state.float_dtype).view(self.state.batch_size, self.beam_size, -1)
                * self.beam_alpha
            )
            self.state.init_batch_lm_states_candidates = init_batch_lm_states_candidates.view(
                self.state.batch_size, self.beam_size, -1
            )

            self.state.batch_lm_states = self.state.init_batch_lm_states.clone()
            self.state.batch_lm_states_candidates = self.state.init_batch_lm_states_candidates.clone()
            self.state.lm_scores[..., :-1].copy_(self.state.init_lm_scores.view(self.state.batch_size, self.state.beam_size, -1))
            self.state.batch_lm_states_prev = self.state.init_batch_lm_states.clone()

        if self.cuda_graphs_mode is self.CudaGraphsMode.FULL_GRAPH:
            self._full_graph_compile()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_WHILE_LOOPS:
            self._partial_graphs_compile()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_GRAPHS:
            # no graphs needed
            pass
        else:
            raise NotImplementedError

    def _partial_graphs_compile(self):
        """Compile decoding by parts"""
        # Always create a new stream, because the per-thread default stream disallows stream capture to a graph.
        stream_for_graph = torch.cuda.Stream(self.state.device)
        stream_for_graph.wait_stream(torch.cuda.default_stream(self.state.device))
        self.separate_graphs = SeparateGraphsBatchedBeamCTC()
        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs._before_process_batch, stream=stream_for_graph, capture_error_mode="thread_local"
            ),
        ):
            self._before_process_batch()

        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs._process_batch, stream=stream_for_graph, capture_error_mode="thread_local"
            ),
        ):
            self._process_batch()

        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs._after_process_batch, stream=stream_for_graph, capture_error_mode="thread_local"
            ),
        ):
            self._after_process_batch()

    def _full_graph_compile(self):
        """Compile full graph for decoding"""
        # Always create a new stream, because the per-thread default stream disallows stream capture to a graph.
        stream_for_graph = torch.cuda.Stream(self.state.device)
        self.full_graph = torch.cuda.CUDAGraph()

        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(self.full_graph, stream=stream_for_graph, capture_error_mode="thread_local"),
        ):
            self._before_process_batch()
            capture_status, _, graph, _, _ = cu_call(
                cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream(device=self.state.device).cuda_stream)
            )

            assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

            # capture: while self.active_mask_any:
            (loop_conditional_handle,) = cu_call(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
            loop_kernel = self._create_process_batch_kernel()
            active_mask_any_ptr = np.array([self.state.active_mask_any.data_ptr()], dtype=np.uint64)
            loop_args = np.array(
                [loop_conditional_handle.getPtr(), active_mask_any_ptr.ctypes.data],
                dtype=np.uint64,
            )
            # loop while there are active utterances
            with with_conditional_node(loop_kernel, loop_args, loop_conditional_handle, device=self.state.device):
                self._process_batch()
            
            self._after_process_batch()

    def _before_process_batch(self):
        """
        Clears state and compute initial active mask
        """
        # torch.cuda.set_sync_debug_mode(2)
        self.state.batched_hyps.clear_()

        # last found labels - initially <SOS> (<blank>) symbol
        self.state.next_scores.fill_(0.0)
        self.state.next_labels.fill_(0.0)
        self.state.next_indices.fill_(0.0)
        self.state.total_scores.fill_(INACTIVE_SCORE)
        self.state.curr_length.fill_(0)
        self.state.log_probs.fill_(0.0)
        self.state.lm_scores.fill_(0.0)
        self.state.repeated_mask.fill_(0)
        self.state.repeated_or_blank_mask.fill_(0)
        
        torch.sub(self.state.decoder_output_lengths, 1, out=self.state.last_timesteps)

        # masks for utterances in batch
        # same as: active_mask = self.encoder_output_length > 0
        torch.greater(self.state.decoder_output_lengths, 0, out=self.state.active_mask)

        # same as: self.active_mask_any = active_mask.any()
        torch.any(self.state.active_mask, out=self.state.active_mask_any)
        # torch.cuda.set_sync_debug_mode(0)
        
        if self.ngram_lm_batch is not None:
            device = self.state.device
            self.ngram_lm_batch.to(device)

            self.state.init_batch_lm_states = self.ngram_lm_batch.get_init_states(
                batch_size=self.state.batch_size * self.beam_size, bos=True
            ).view(self.state.batch_size, self.beam_size)
            init_lm_scores, init_batch_lm_states_candidates = self.ngram_lm_batch.advance(
                states=self.state.init_batch_lm_states.view(-1)
            )  # vocab_size_no_blank
            self.state.init_lm_scores = (
                init_lm_scores.to(dtype=self.state.float_dtype).view(self.state.batch_size, self.beam_size, -1)
                * self.beam_alpha
            )
            self.state.init_batch_lm_states_candidates = init_batch_lm_states_candidates.view(
                self.state.batch_size, self.beam_size, -1
            )

            self.state.batch_lm_states = self.state.init_batch_lm_states.clone()
            self.state.batch_lm_states_candidates = self.state.init_batch_lm_states_candidates.clone()
            self.state.lm_scores[..., :-1].copy_(self.state.init_lm_scores.view(self.state.batch_size, self.state.beam_size, -1))
            self.state.batch_lm_states_prev = self.state.init_batch_lm_states.clone()
        
    def _process_batch(self): 
        # torch.cuda.set_sync_debug_mode(2)
        log_probs = (
            self.state.decoder_outputs
            .index_select(dim=1, index=self.state.curr_length.repeat(self.beam_size))
        )
        
        # print("bstbstbst", self.beam_size_token)
        # if self.beam_size_token is not None:
        #     _, topk_idx = log_probs.topk(k=self.beam_size_token, largest=True, sorted=True)
        #     mask = self.state.false_mask.scatter_(dim=2, index=topk_idx, value=True)
        #     log_probs.masked_fill_(~mask, float('-inf'))
        
        if self.ngram_lm_batch is not None:
            lm_scores, batch_lm_states_candidates = self.ngram_lm_batch.advance(states=self.state.batch_lm_states.view(-1))

            self.state.batch_lm_states_candidates.copy_(batch_lm_states_candidates.view(self.state.batch_size, self.state.beam_size, -1))
            log_probs[..., :-1] += self.beam_alpha * lm_scores.view(self.state.batch_size, self.state.beam_size, -1)
        
        self.state.log_probs.copy_(log_probs).mul_(self.state.log10e)            
        self.state.log_probs.add_(self.state.batched_hyps.scores[:, :, None])
        
        torch.eq(self.state.batched_hyps.last_label[:, :, None], self.state.vocab[None, None, :], out=self.state.repeated_mask)
        torch.logical_or(self.state.blank_mask[None, None, :], self.state.repeated_mask, out=self.state.repeated_or_blank_mask)
        torch.where(self.state.repeated_or_blank_mask, self.state.log_probs, self.state.log_probs + self.beam_beta, out=self.state.log_probs)
        
        hyps_scores, hyps_candidates_indices = torch.topk(self.state.log_probs.view(self.state.batch_size, -1), k=self.beam_size, largest=True, sorted=True)
        
        self.state.next_scores.copy_(hyps_scores)
        self.state.next_indices.copy_(hyps_candidates_indices // self.state.vocab_size)
        self.state.next_labels.copy_(torch.gather(self.state.batch_labels, dim=-1, index=(hyps_candidates_indices % self.state.vocab_size).unsqueeze(-1)).squeeze(2))
        self.state.next_scores.view(self.state.batch_size, -1)[
            self.state.next_scores.view(self.state.batch_size, -1) <= 
            self.state.next_scores.view(self.state.batch_size, -1).max(dim=-1, keepdim=True).values - 
            self.beam_threshold
        ] = float('-inf')
        
        torch.where(self.state.active_mask, self.state.next_labels, self.state.MINUS_ONE_TENSOR, out=self.state.next_labels)
        
        if self.ngram_lm_batch is not None:
            repeating_mask = self.state.next_labels == torch.gather(self.state.batched_hyps.last_label, dim=-1, index=self.state.next_indices)
            blank_mask = self.state.next_labels == self._blank_index
            preserve_state_mask = repeating_mask | blank_mask | ~ self.state.active_mask
            
            next_labels_masked = torch.where(blank_mask | ~ self.state.active_mask, 0, self.state.next_labels)
            # batch_lm_states: [(BxBeam)]
            # batch_lm_states_candidates: [(BxBeam) x V (without blank)]
            self.state.batch_lm_states_candidates.copy_(
                torch.gather(
                    self.state.batch_lm_states_candidates,
                    dim=1,
                    index=self.state.next_indices[:, :, None].expand(
                        self.state.batch_size, self.beam_size, self.state.batch_lm_states_candidates.shape[-1]
                        ),
                )
            )
            self.state.batch_lm_states_prev.copy_(
                torch.gather(
                    self.state.batch_lm_states, 
                    dim=1, 
                    index=self.state.next_indices
                )
            )

            self.state.batch_lm_states.copy_(
                torch.gather(
                    self.state.batch_lm_states_candidates, 
                    dim=-1, 
                    index=next_labels_masked.unsqueeze(-1)
                ).squeeze()
            )
            self.state.batch_lm_states.copy_(
                torch.where(
                    preserve_state_mask, 
                    self.state.batch_lm_states_prev, 
                    self.state.batch_lm_states
                )
            )
        
        self.state.batched_hyps.add_results_(self.state.next_indices, self.state.next_labels, self.state.next_scores)
        self.state.batched_hyps.self_recombine_hyps_()
        
        self.state.curr_length.add_(1)
        torch.greater(self.state.last_timesteps, self.state.curr_length, out=self.state.active_mask)
        torch.any(self.state.active_mask, out=self.state.active_mask_any)
        
        # torch.cuda.set_sync_debug_mode(0)
  
    def _after_process_batch(self):     
        if self.ngram_lm_batch is not None:
            self.state.batched_hyps.scores += self.ngram_lm_batch.get_final(self.state.batch_lm_states).view(self.state.batch_size, self.beam_size) * np.log10(np.e) * self.beam_alpha

    def __call__(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
    ) -> Tuple[rnnt_utils.BatchedHyps, Optional[rnnt_utils.BatchedAlignments], Any]:
        if self.cuda_graphs_mode is not None and x.device.type == "cuda":
            return self.batched_beam_search_cuda_graphs(decoder_outputs=x, decoder_output_lengths=out_len)

        return self.batched_beam_search_torch(decoder_outputs=x, decoder_output_lengths=out_len)