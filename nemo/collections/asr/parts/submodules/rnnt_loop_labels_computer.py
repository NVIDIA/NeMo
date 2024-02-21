# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import contextlib
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodMixin
from nemo.core.utils.cuda_python_utils import (
    assert_drv,
    check_cuda_python_cuda_graphs_conditional_nodes_supported,
    cu_call,
)
from nemo.utils import logging

try:
    from cuda import cuda, cudart, nvrtc

    HAVE_CUDA_PYTHON = True
except ImportError:
    HAVE_CUDA_PYTHON = False


def run_nvrtc(kernel_string, kernel_name):
    err, prog = nvrtc.nvrtcCreateProgram(str.encode(kernel_string), b"while_loop_labels_conditional.cu", 0, [], [])
    assert_drv(err)
    # Compile program
    # Not specifying --gpu-architecture will default us to a fairly low compute capability, which is a safe bet.
    # Otherwise, there are ways to query the current device's compute capability.
    # https://stackoverflow.com/questions/48283009/nvcc-get-device-compute-capability-in-runtime
    opts = []
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    err, size = nvrtc.nvrtcGetProgramLogSize(prog)
    buf = b" " * size
    (err,) = nvrtc.nvrtcGetProgramLog(prog, buf)
    assert_drv(err)

    # Get PTX from compilation
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    assert_drv(err)
    ptx = b" " * ptxSize
    (err,) = nvrtc.nvrtcGetPTX(prog, ptx)
    assert_drv(err)

    ptx = np.char.array(ptx)
    err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
    assert_drv(err)
    err, kernel = cuda.cuModuleGetFunction(module, kernel_name)
    assert_drv(err)

    return kernel


def create_outer_while_loop_kernel():
    """
    Creates a kernel that evaluates whether or not to enter the for loop body.
    Effectively substitutes for `for time_idx in range(trip_count)`
    such that that for loop can run on a GPU.
    """
    kernel_string = r"""\
    typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;

    extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);

    extern "C" __global__
    void outer_loop_labels_conditional(cudaGraphConditionalHandle handle, const bool *active_mask_any)
    {
     cudaGraphSetConditional(handle, *active_mask_any);
    }
    """
    return run_nvrtc(kernel_string, b"outer_loop_labels_conditional")


def create_inner_while_loop_kernel():
    """
    Evaluates whether or not to keep evaluating the inner while loop body.
    Continue until all elements of the batch output blank or the while loop
    has run max_symbols times.
    """
    kernel_string = r"""\
    typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;

    extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);

    extern "C" __global__
    void inner_find_non_blank_conditional(cudaGraphConditionalHandle handle, const bool *advance_mask_any)
    {
     cudaGraphSetConditional(handle, *advance_mask_any);
    }
    """
    return run_nvrtc(kernel_string, b"inner_find_non_blank_conditional")


@contextlib.contextmanager
def with_conditional_node(while_loop_kernel, while_loop_args, while_loop_conditional_handle, device):
    """
    Even though we add a conditional node only once, we need to
    capture the kernel that calls cudaGraphSetConditional() both
    before in the parent graph containing the while loop body graph
    and after the rest of the while loop body graph (because we need
    to decide both whether to enter the loop, and also whether to
    execute the next iteration of the loop).
    """
    capture_status, _, graph, _, _ = cu_call(cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream().cuda_stream))
    assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

    cuda.cuLaunchKernel(
        while_loop_kernel, 1, 1, 1, 1, 1, 1, 0, torch.cuda.current_stream().cuda_stream, while_loop_args.ctypes.data, 0
    )

    capture_status, _, graph, dependencies, _ = cu_call(
        cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream().cuda_stream)
    )
    assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

    driver_params = cuda.CUgraphNodeParams()
    driver_params.type = cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
    driver_params.conditional.handle = while_loop_conditional_handle
    driver_params.conditional.type = cuda.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_WHILE
    driver_params.conditional.size = 1
    # Work around until https://github.com/NVIDIA/cuda-python/issues/55 is fixed
    driver_params.conditional.phGraph_out = [cuda.CUgraph()]
    (ctx,) = cu_call(cuda.cuCtxGetCurrent())
    driver_params.conditional.ctx = ctx

    # Use driver API here because of bug in cuda-python runtime API: https://github.com/NVIDIA/cuda-python/issues/55
    # TODO: Change call to this after fix goes in:
    # node, = cu_call(cudart.cudaGraphAddNode(graph, dependencies, len(dependencies), driver_params))
    (node,) = cu_call(cuda.cuGraphAddNode(graph, dependencies, len(dependencies), driver_params))
    body_graph = driver_params.conditional.phGraph_out[0]

    cu_call(
        cudart.cudaStreamUpdateCaptureDependencies(
            torch.cuda.current_stream().cuda_stream,
            [node],
            1,
            cudart.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies,
        )
    )
    body_stream = torch.cuda.Stream(device=device)
    previous_stream = torch.cuda.current_stream(device=device)
    cu_call(
        cudart.cudaStreamBeginCaptureToGraph(
            body_stream.cuda_stream,
            body_graph,
            None,
            None,
            0,
            cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal,
        )
    )
    torch.cuda.set_stream(body_stream)

    yield body_stream, body_graph

    cuda.cuLaunchKernel(
        while_loop_kernel, 1, 1, 1, 1, 1, 1, 0, body_stream.cuda_stream, while_loop_args.ctypes.data, 0
    )

    cudart.cudaStreamEndCapture(body_stream.cuda_stream)

    torch.cuda.set_stream(previous_stream)


class GreedyBatchedRNNTLoopLabelsComputer(ConfidenceMethodMixin):
    """
    Loop Labels algorithm implementation. Callable.
    """

    def __init__(
        self,
        decoder,
        joint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments=False,
        preserve_frame_confidence=False,
        confidence_method_cfg: Optional[DictConfig] = None,
        allow_cuda_graphs: bool = True,
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
        """
        super().__init__()
        self.decoder = decoder
        self.joint = joint
        self._blank_index = blank_index
        self.max_symbols = max_symbols_per_step
        self.preserve_alignments = preserve_alignments
        self.preserve_frame_confidence = preserve_frame_confidence
        self._SOS = self._blank_index
        self._init_confidence_method(confidence_method_cfg=confidence_method_cfg)
        assert self._SOS == self._blank_index  # "blank as pad" algorithm only

        self.use_cuda_graphs = allow_cuda_graphs

        if self.use_cuda_graphs and self.max_symbols is None:
            logging.warning("Max symbols is None, which is not allowed with Cuda graphs.")
            self.use_cuda_graphs = False

        if self.use_cuda_graphs:
            try:
                check_cuda_python_cuda_graphs_conditional_nodes_supported()
            except ImportError as e:
                logging.warning(f"No conditional node support. Cuda graphs will be disabled,\n{e.msg}")
                self.use_cuda_graphs = False

        if self.use_cuda_graphs:
            self._init_for_cuda_graphs()

    def _init_for_cuda_graphs(self):
        # Reasonable default maximum time. 375 frames * (80ms / frame) = 30 seconds
        # 80ms is the frame size of recent fastconformer models
        # This does not affect correctness.
        self.max_time = 375
        self.batch_size = 0
        self.graph = None
        self.first_call = True
        self.cuda_device = torch.device("cuda")
        self.encoder_output_projected = None
        self.encoder_output_length = None
        self.batched_hyps = None

    def loop_labels_torch(
        self, x: torch.Tensor, out_len: torch.Tensor,
    ) -> Tuple[rnnt_utils.BatchedHyps, Optional[rnnt_utils.BatchedAlignments], Any]:
        """
        Optimized batched greedy decoding.
        Iterates over labels, on each step finding the next non-blank label
        (evaluating Joint multiple times in inner loop); It uses a minimal possible amount of calls
        to prediction network (with maximum possible batch size),
        which makes it especially useful for scaling the prediction network.
        During decoding all active hypotheses ("texts") have the same lengths.

        Args:
            x: output from the encoder
            out_len: lengths of the utterances in `x`
        """
        batch_size, max_time, _unused = x.shape
        device = x.device

        x = self.joint.project_encoder(x)  # do not recalculate joint projection, project only once

        # init output structures: BatchedHyps (for results), BatchedAlignments + last decoder state
        # init empty batched hypotheses
        batched_hyps = rnnt_utils.BatchedHyps(
            batch_size=batch_size,
            init_length=max_time * self.max_symbols if self.max_symbols is not None else max_time,
            device=x.device,
            float_dtype=x.dtype,
        )
        # sample state, will be replaced further when the decoding for hypothesis is done
        last_decoder_state = self.decoder.initialize_state(x)
        # init alignments if necessary
        use_alignments = self.preserve_alignments or self.preserve_frame_confidence
        # always use alignments variable - for torch.jit adaptation, but keep it as minimal as possible
        alignments = rnnt_utils.BatchedAlignments(
            batch_size=batch_size,
            logits_dim=self.joint.num_classes_with_blank,
            init_length=max_time * 2 if use_alignments else 1,  # blank for each timestep + text tokens
            device=x.device,
            float_dtype=x.dtype,
            store_alignments=self.preserve_alignments,
            store_frame_confidence=self.preserve_frame_confidence,
        )

        # initial state, needed for torch.jit to compile (cannot handle None)
        state = self.decoder.initialize_state(x)
        # indices of elements in batch (constant)
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
        # last found labels - initially <SOS> (<blank>) symbol
        labels = torch.full_like(batch_indices, fill_value=self._SOS)

        # time indices
        time_indices = torch.zeros_like(batch_indices)
        safe_time_indices = torch.zeros_like(time_indices)  # time indices, guaranteed to be < out_len
        time_indices_current_labels = torch.zeros_like(time_indices)
        last_timesteps = out_len - 1

        # masks for utterances in batch
        active_mask: torch.Tensor = out_len > 0
        advance_mask = torch.empty_like(active_mask)

        # for storing the last state we need to know what elements became "inactive" on this step
        active_mask_prev = torch.empty_like(active_mask)
        became_inactive_mask = torch.empty_like(active_mask)

        # loop while there are active utterances
        first_step = True
        while active_mask.any():
            active_mask_prev.copy_(active_mask, non_blocking=True)
            # stage 1: get decoder (prediction network) output
            if first_step:
                # start of the loop, SOS symbol is passed into prediction network, state is None
                # we need to separate this for torch.jit
                decoder_output, state, *_ = self.decoder.predict(
                    labels.unsqueeze(1), None, add_sos=False, batch_size=batch_size
                )
                first_step = False
            else:
                decoder_output, state, *_ = self.decoder.predict(
                    labels.unsqueeze(1), state, add_sos=False, batch_size=batch_size
                )
            decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection

            # stage 2: get joint output, iteratively seeking for non-blank labels
            # blank label in `labels` tensor means "end of hypothesis" (for this index)
            logits = (
                self.joint.joint_after_projection(x[batch_indices, safe_time_indices].unsqueeze(1), decoder_output,)
                .squeeze(1)
                .squeeze(1)
            )
            scores, labels = logits.max(-1)

            # search for non-blank labels using joint, advancing time indices for blank labels
            # checking max_symbols is not needed, since we already forced advancing time indices for such cases
            blank_mask = labels == self._blank_index
            time_indices_current_labels.copy_(time_indices, non_blocking=True)
            if use_alignments:
                if self.preserve_frame_confidence:
                    logits = F.log_softmax(logits, dim=-1)
                alignments.add_results_masked_(
                    active_mask=active_mask,
                    time_indices=time_indices_current_labels,
                    logits=logits if self.preserve_alignments else None,
                    labels=labels if self.preserve_alignments else None,
                    confidence=self._get_confidence_tensor(logits) if self.preserve_frame_confidence else None,
                )

            # advance_mask is a mask for current batch for searching non-blank labels;
            # each element is True if non-blank symbol is not yet found AND we can increase the time index
            time_indices += blank_mask
            torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
            torch.less(time_indices, out_len, out=active_mask)
            torch.logical_and(active_mask, blank_mask, out=advance_mask)

            # inner loop: find next non-blank labels (if exist)
            while advance_mask.any():
                # same as: time_indices_current_labels[advance_mask] = time_indices[advance_mask], but non-blocking
                # store current time indices to use further for storing the results
                torch.where(advance_mask, time_indices, time_indices_current_labels, out=time_indices_current_labels)
                logits = (
                    self.joint.joint_after_projection(
                        x[batch_indices, safe_time_indices].unsqueeze(1), decoder_output,
                    )
                    .squeeze(1)
                    .squeeze(1)
                )
                # get labels (greedy) and scores from current logits, replace labels/scores with new
                # labels[advance_mask] are blank, and we are looking for non-blank labels
                more_scores, more_labels = logits.max(-1)
                # same as: labels[advance_mask] = more_labels[advance_mask], but non-blocking
                torch.where(advance_mask, more_labels, labels, out=labels)
                # same as: scores[advance_mask] = more_scores[advance_mask], but non-blocking
                torch.where(advance_mask, more_scores, scores, out=scores)

                if use_alignments:
                    if self.preserve_frame_confidence:
                        logits = F.log_softmax(logits, dim=-1)
                    alignments.add_results_masked_(
                        active_mask=advance_mask,
                        time_indices=time_indices_current_labels,
                        logits=logits if self.preserve_alignments else None,
                        labels=more_labels if self.preserve_alignments else None,
                        confidence=self._get_confidence_tensor(logits) if self.preserve_frame_confidence else None,
                    )

                blank_mask = labels == self._blank_index
                time_indices += blank_mask
                torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
                torch.less(time_indices, out_len, out=active_mask)
                torch.logical_and(active_mask, blank_mask, out=advance_mask)

            # stage 3: filter labels and state, store hypotheses
            # select states for hyps that became inactive (is it necessary?)
            # this seems to be redundant, but used in the `loop_frames` output
            torch.ne(active_mask, active_mask_prev, out=became_inactive_mask)
            self.decoder.batch_replace_states_mask(
                src_states=state, dst_states=last_decoder_state, mask=became_inactive_mask,
            )

            # store hypotheses
            if self.max_symbols is not None:
                # pre-allocated memory, no need for checks
                batched_hyps.add_results_masked_no_checks_(
                    active_mask, labels, time_indices_current_labels, scores,
                )
            else:
                # auto-adjusted storage
                batched_hyps.add_results_masked_(
                    active_mask, labels, time_indices_current_labels, scores,
                )

            # stage 4: to avoid looping, go to next frame after max_symbols emission
            if self.max_symbols is not None:
                # if labels are non-blank (not end-of-utterance), check that last observed timestep with label:
                # if it is equal to the current time index, and number of observations is >= max_symbols, force blank
                force_blank_mask = torch.logical_and(
                    active_mask,
                    torch.logical_and(
                        torch.logical_and(
                            labels != self._blank_index, batched_hyps.last_timestep_lasts >= self.max_symbols,
                        ),
                        batched_hyps.last_timestep == time_indices,
                    ),
                )
                time_indices += force_blank_mask  # emit blank => advance time indices
                # update safe_time_indices, non-blocking
                torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
                # same as: active_mask = time_indices < out_len
                torch.less(time_indices, out_len, out=active_mask)
        if use_alignments:
            return batched_hyps, alignments, last_decoder_state
        return batched_hyps, None, last_decoder_state

    def _graph_reinitialize(
        self,
        max_time: int,
        batch_size: int,
        encoder_output_projected: torch.Tensor,
        encoder_output_length: torch.Tensor,
    ):
        logging.warning(f"Reinit {self.max_time} -> {max_time}, {self.batch_size} -> {batch_size}")
        self.max_time = max(self.max_time, max_time)
        self.batch_size = max(self.batch_size, batch_size)
        self.cuda_device = encoder_output_projected.device
        float_dtype = encoder_output_projected.dtype

        logging.warning(f"Reinit Graph")
        self.encoder_output_projected = torch.zeros(
            (self.batch_size, self.max_time, encoder_output_projected.shape[-1]),
            dtype=float_dtype,
            device=self.cuda_device,
        )
        self.encoder_output_length = torch.zeros(
            (self.batch_size,), dtype=encoder_output_length.dtype, device=encoder_output_length.device
        )

        self.batched_hyps = rnnt_utils.BatchedHyps(
            batch_size=self.batch_size,
            init_length=self.max_time * self.max_symbols,
            device=self.cuda_device,
            float_dtype=float_dtype,
        )

        self.labels = torch.zeros([self.batch_size], dtype=torch.long, device=self.cuda_device)
        self.scores = torch.zeros([self.batch_size], dtype=float_dtype, device=self.cuda_device)

        # indices of elements in batch (constant)
        self.batch_indices = torch.arange(self.batch_size, dtype=torch.long, device=self.cuda_device)

        self.time_indices = torch.zeros_like(self.batch_indices)
        self.safe_time_indices = torch.zeros_like(self.batch_indices)
        self.time_indices_current_labels = torch.zeros_like(self.time_indices)
        self.last_timesteps = torch.zeros_like(self.time_indices)

        self.active_mask = torch.zeros([self.batch_size], dtype=torch.bool, device=self.cuda_device)
        self.advance_mask = torch.zeros_like(self.active_mask)
        self.blank_mask = torch.zeros_like(self.active_mask)

        self.active_mask_any = torch.tensor(True, device=self.cuda_device, dtype=torch.bool)
        self.advance_mask_any = torch.tensor(True, device=self.cuda_device, dtype=torch.bool)

        self.active_mask_prev = torch.zeros_like(self.active_mask)
        self.became_inactive_mask = torch.zeros_like(self.active_mask)

        self.last_decoder_state = self.decoder.initialize_state(self.encoder_output_projected)
        self.state = self.decoder.initialize_state(self.encoder_output_projected)

        decoder_output, self.state, *_ = self.decoder.predict(
            self.labels.unsqueeze(1), self.state, add_sos=False, batch_size=self.batch_size
        )
        self.decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(torch.cuda.Stream(device=self.cuda_device)), torch.inference_mode(), torch.cuda.graph(
            self.graph
        ):
            self.batched_hyps.clear_()

            # initial state, needed for torch.jit to compile (cannot handle None)
            self.state[0].fill_(0.0)
            self.state[1].fill_(0.0)
            # last found labels - initially <SOS> (<blank>) symbol
            self.labels.fill_(self._SOS)
            self.scores.fill_(0.0)

            # time indices
            # time_indices = torch.zeros_like(batch_indices)
            # safe_time_indices = torch.zeros_like(time_indices)  # time indices, guaranteed to be < out_len
            self.time_indices.fill_(0)
            self.safe_time_indices.fill_(0)
            self.time_indices_current_labels.fill_(0)
            torch.sub(self.encoder_output_length, 1, out=self.last_timesteps)

            # masks for utterances in batch
            # active_mask: torch.Tensor = self.encoder_output_length > 0
            # advance_mask = torch.empty_like(active_mask)
            torch.greater(self.encoder_output_length, 0, out=self.active_mask)

            # for storing the last state we need to know what elements became "inactive" on this step
            # self.active_mask_any = active_mask.any()
            torch.any(self.active_mask, out=self.active_mask_any)

            capture_status, _, graph, _, _ = cu_call(
                cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream(device=self.cuda_device).cuda_stream)
            )
            assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

            (outer_loop_conditional_handle,) = cu_call(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
            outer_loop_kernel = create_outer_while_loop_kernel()
            active_mask_any_ptr = np.array([self.active_mask_any.data_ptr()], dtype=np.uint64)
            outer_loop_args = np.array(
                [outer_loop_conditional_handle.getPtr(), active_mask_any_ptr.ctypes.data], dtype=np.uint64,
            )

            # loop while there are active utterances
            # while self.active_mask_any:
            with with_conditional_node(
                outer_loop_kernel, outer_loop_args, outer_loop_conditional_handle, device=self.cuda_device
            ):
                self._before_inner_loop()
                inner_while_loop_kernel = create_inner_while_loop_kernel()
                (inner_loop_conditional_handle,) = cu_call(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
                advance_mask_any_ptr = np.array([self.advance_mask_any.data_ptr()], dtype=np.uint64)
                inner_loop_args = np.array(
                    [inner_loop_conditional_handle.getPtr(), advance_mask_any_ptr.ctypes.data,], dtype=np.uint64,
                )
                # while self.advance_mask_any.item():

                with with_conditional_node(
                    inner_while_loop_kernel, inner_loop_args, inner_loop_conditional_handle, device=self.cuda_device
                ):
                    self._inner_loop()
                self._after_inner_loop()

    def _before_inner_loop(self):
        self.active_mask_prev.copy_(self.active_mask, non_blocking=True)
        # stage 1: get decoder (prediction network) output
        decoder_output, state, *_ = self.decoder.predict(
            self.labels.unsqueeze(1), self.state, add_sos=False, batch_size=self.batch_size
        )
        self.state[0].copy_(state[0])
        self.state[1].copy_(state[1])
        decoder_output_projected = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection
        self.decoder_output.copy_(decoder_output_projected)

        # stage 2: get joint output, iteratively seeking for non-blank labels
        # blank label in `labels` tensor means "end of hypothesis" (for this index)
        logits = (
            self.joint.joint_after_projection(
                self.encoder_output_projected[self.batch_indices, self.safe_time_indices].unsqueeze(1),
                self.decoder_output,
            )
            .squeeze(1)
            .squeeze(1)
        )
        # scores, labels = logits.max(-1)
        torch.max(logits, dim=-1, out=(self.scores, self.labels))

        # search for non-blank labels using joint, advancing time indices for blank labels
        # checking max_symbols is not needed, since we already forced advancing time indices for such cases
        torch.eq(self.labels, self._blank_index, out=self.blank_mask)
        # blank_mask = self.labels == self._blank_index
        self.time_indices_current_labels.copy_(self.time_indices, non_blocking=True)
        # if use_alignments:
        #     if self.preserve_frame_confidence:
        #         logits = F.log_softmax(logits, dim=-1)
        #     alignments.add_results_masked_(
        #         active_mask=active_mask,
        #         time_indices=time_indices_current_labels,
        #         logits=logits if self.preserve_alignments else None,
        #         labels=labels if self.preserve_alignments else None,
        #         confidence=self._get_confidence_tensor(logits) if self.preserve_frame_confidence else None,
        #     )

        # advance_mask is a mask for current batch for searching non-blank labels;
        # each element is True if non-blank symbol is not yet found AND we can increase the time index
        # self.time_indices += self.blank_mask
        # self.time_indices = self.time_indices + self.blank_mask
        self.time_indices.add_(self.blank_mask)
        torch.minimum(self.time_indices, self.last_timesteps, out=self.safe_time_indices)
        torch.less(self.time_indices, self.encoder_output_length, out=self.active_mask)
        torch.logical_and(self.active_mask, self.blank_mask, out=self.advance_mask)

        # inner loop: find next non-blank labels (if exist)

        # self.advance_mask_any = advance_mask.any()
        torch.any(self.advance_mask, out=self.advance_mask_any)

    def _inner_loop(self):
        # same as: time_indices_current_labels[advance_mask] = time_indices[advance_mask], but non-blocking
        # store current time indices to use further for storing the results
        torch.where(
            self.advance_mask,
            self.time_indices,
            self.time_indices_current_labels,
            out=self.time_indices_current_labels,
        )
        logits = (
            self.joint.joint_after_projection(
                self.encoder_output_projected[self.batch_indices, self.safe_time_indices].unsqueeze(1),
                self.decoder_output,
            )
            .squeeze(1)
            .squeeze(1)
        )
        # get labels (greedy) and scores from current logits, replace labels/scores with new
        # labels[advance_mask] are blank, and we are looking for non-blank labels
        more_scores, more_labels = logits.max(-1)
        # same as: labels[advance_mask] = more_labels[advance_mask], but non-blocking
        torch.where(self.advance_mask, more_labels, self.labels, out=self.labels)
        # same as: scores[advance_mask] = more_scores[advance_mask], but non-blocking
        torch.where(self.advance_mask, more_scores, self.scores, out=self.scores)

        # if use_alignments:
        #     if self.preserve_frame_confidence:
        #         logits = F.log_softmax(logits, dim=-1)
        #     alignments.add_results_masked_(
        #         active_mask=advance_mask,
        #         time_indices=time_indices_current_labels,
        #         logits=logits if self.preserve_alignments else None,
        #         labels=more_labels if self.preserve_alignments else None,
        #         confidence=self._get_confidence_tensor(logits) if self.preserve_frame_confidence else None,
        #     )

        # blank_mask = self.labels == self._blank_index
        torch.eq(self.labels, self._blank_index, out=self.blank_mask)
        # self.time_indices += self.blank_mask
        self.time_indices.add_(self.blank_mask)

        torch.minimum(self.time_indices, self.last_timesteps, out=self.safe_time_indices)
        torch.less(self.time_indices, self.encoder_output_length, out=self.active_mask)
        torch.logical_and(self.active_mask, self.blank_mask, out=self.advance_mask)
        torch.any(self.advance_mask, out=self.advance_mask_any)

    def _after_inner_loop(self):
        # stage 3: filter labels and state, store hypotheses
        # select states for hyps that became inactive (is it necessary?)
        # this seems to be redundant, but used in the `loop_frames` output
        torch.ne(self.active_mask, self.active_mask_prev, out=self.became_inactive_mask)
        self.decoder.batch_replace_states_mask(
            src_states=self.state, dst_states=self.last_decoder_state, mask=self.became_inactive_mask,
        )

        # store hypotheses
        # if self.max_symbols is not None:
        #     # pre-allocated memory, no need for checks
        #     self.batched_hyps.add_results_masked_no_checks_(
        #         active_mask, labels, time_indices_current_labels, scores,
        #     )
        # else:
        # auto-adjusted storage
        self.batched_hyps.add_results_masked_no_checks_(
            self.active_mask, self.labels, self.time_indices_current_labels, self.scores,
        )

        # stage 4: to avoid looping, go to next frame after max_symbols emission
        if self.max_symbols is not None:
            # if labels are non-blank (not end-of-utterance), check that last observed timestep with label:
            # if it is equal to the current time index, and number of observations is >= max_symbols, force blank
            force_blank_mask = torch.logical_and(
                self.active_mask,
                torch.logical_and(
                    torch.logical_and(
                        self.labels != self._blank_index, self.batched_hyps.last_timestep_lasts >= self.max_symbols,
                    ),
                    self.batched_hyps.last_timestep == self.time_indices,
                ),
            )
            self.time_indices.add_(force_blank_mask)  # emit blank => advance time indices
            # update safe_time_indices, non-blocking
            torch.minimum(self.time_indices, self.last_timesteps, out=self.safe_time_indices)
            # same as: active_mask = time_indices < out_len
            torch.less(self.time_indices, self.encoder_output_length, out=self.active_mask)
        torch.any(self.active_mask, out=self.active_mask_any)

    def __call__(
        self, x: torch.Tensor, out_len: torch.Tensor,
    ) -> Tuple[rnnt_utils.BatchedHyps, Optional[rnnt_utils.BatchedAlignments], Any]:
        if not self.use_cuda_graphs or self.preserve_alignments or self.preserve_frame_confidence:
            return self.loop_labels_torch(x=x, out_len=out_len)

        # if self.preserve_alignments:
        #     raise NotImplementedError("`preserve_alignments` support is not available with cuda graphs (but could be)")
        #
        # if self.preserve_frame_confidence:
        #     raise NotImplementedError(
        #         "`preserve_frame_confidence` support is not available with cuda graphs (but could be)"
        #     )

        x = self.joint.project_encoder(x)  # do not recalculate joint projection, project only once
        batch_size = x.shape[0]
        max_time = x.shape[1]

        if torch.is_autocast_enabled():
            x = x.to(torch.get_autocast_gpu_dtype())

        if max_time > self.max_time or batch_size > self.batch_size:
            self._graph_reinitialize(max_time, batch_size, x, out_len)
        self.encoder_output_length.fill_(0)
        self.encoder_output_projected[: x.shape[0], : x.shape[1], ...].copy_(x)
        self.encoder_output_length[: out_len.shape[0]].copy_(out_len)
        self.graph.replay()

        return self.batched_hyps, None, self.last_decoder_state
