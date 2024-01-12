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

from nemo.core.utils.cuda_python_utils import check_cuda_python_cuda_graphs_conditional_nodes_supported

check_cuda_python_cuda_graphs_conditional_nodes_supported()

import contextlib
import ctypes
import time
from dataclasses import dataclass, field
from itertools import product
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from cuda import cuda, cudart, nvrtc
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.modules import rnnt_abstract
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodConfig, ConfidenceMethodMixin
from nemo.collections.common.parts.rnn import label_collate
from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import AcousticEncodedRepresentation, ElementType, HypothesisType, LengthsType, NeuralType
from nemo.utils import logging

def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


def cu_call(f_call_out):
    error, *others = f_call_out
    if error != cudart.cudaError_t.cudaSuccess:
        # import ipdb; ipdb.set_trace()
        raise Exception(f"CUDA failure! {error}")
    else:
        # print("GALVEZ:", others)
        return tuple(others)


def run_nvrtc(kernel_string, kernel_name):
    err, prog = nvrtc.nvrtcCreateProgram(str.encode(kernel_string), b"while_loop_conditional.cu", 0, [], [])

    ASSERT_DRV(err)

    # Compile program
    # Not specifying --gpu-architecture will default us to a fairly low compute capability, which is a safe bet.
    # Otherwise, there are ways to query the current device's compute capability. 
    # https://stackoverflow.com/questions/48283009/nvcc-get-device-compute-capability-in-runtime
    opts = []
    (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    err, size = nvrtc.nvrtcGetProgramLogSize(prog)
    buf = b" " * size
    (err,) = nvrtc.nvrtcGetProgramLog(prog, buf)
    # print(buf.decode("utf-8"))
    ASSERT_DRV(err)

    # Get PTX from compilation
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    ASSERT_DRV(err)
    ptx = b" " * ptxSize
    (err,) = nvrtc.nvrtcGetPTX(prog, ptx)
    ASSERT_DRV(err)

    # print("GALVEZ:PTX=")
    # print(ptx.decode("utf-8"))
    ptx = np.char.array(ptx)
    err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
    ASSERT_DRV(err)
    err, kernel = cuda.cuModuleGetFunction(module, kernel_name)
    ASSERT_DRV(err)

    return kernel


def create_outer_for_loop_kernel():
    kernel_string = r"""\
    typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;

    extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);

    extern "C" __global__
    void for_loop_conditional(cudaGraphConditionalHandle handle, const long *time_idx, const long *trip_count)
    {
     cudaGraphSetConditional(handle, *time_idx < *trip_count);
    }
    """
    return run_nvrtc(kernel_string, b"for_loop_conditional")


# Observations: If cudaGraphSetConditional is true once, the kernel never completes...
# The GPU is doing *something*. I'm just not sure what...
# No way to query that at runtime...


def create_while_loop_kernel():
    kernel_string = r"""\
    typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;

    extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);

    extern "C" __global__
    void while_loop_conditional(cudaGraphConditionalHandle handle, const bool *not_blank, const long *symbols_added, const long *max_symbols)
    {
     cudaGraphSetConditional(handle, *not_blank && *symbols_added < *max_symbols);
    }
    """
    return run_nvrtc(kernel_string, b"while_loop_conditional")


@contextlib.contextmanager
def with_conditional_node(while_loop_kernel, while_loop_args, while_loop_conditional_handle):
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
    body_stream = torch.cuda.Stream()
    previous_stream = torch.cuda.current_stream()
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


class RNNTGreedyDecodeFast:
    def __init__(self, max_symbols: int, cuda_device, caller):
        assert max_symbols is not None

        self.symbols_added_t = torch.tensor(0, dtype=torch.int64, device=cuda_device)
        self.max_symbols_t = torch.tensor(max_symbols, dtype=torch.int64, device=cuda_device)
        self.max_symbols = max_symbols
        self.not_all_blank_t = torch.tensor(True, dtype=torch.bool, device=cuda_device)

        self.cuda_device = cuda_device

        self.time_idx_t = torch.tensor(0, dtype=torch.int64, device=self.cuda_device)
        self.max_out_len_t = torch.tensor(0, dtype=torch.int64, device=self.cuda_device)

        self.encoder_output = None
        self.encoder_output_length = None
        self.f = None

        # Reasonable default maximum time. 375 frames * 40ms / frame = 15 seconds
        # This affects only the size of the CPU-pinned memory buffers
        self.max_time = 375
        self.batch_size = 0

        self.scores_cpu = None
        self.labels_cpu = None
        self.symbols_per_time_step_cpu = None
        self.graph = None
        self.graph_exec = None

        self.caller = caller

    def _reinitialize(self, max_time, batch_size, encoder_output, encoder_output_length):
        # We need to call _greedy_decode_blank_as_pad at least once
        # before hand in order to make sure that pytorch is
        # "initialize".
        self.caller._greedy_decode_blank_as_pad(encoder_output,
                                                encoder_output_length,
                                                encoder_output.device)

        self.max_time = max(self.max_time, max_time)
        self.batch_size = max(self.batch_size, batch_size)

        self.encoder_output = torch.zeros(
            (self.batch_size, self.max_time, encoder_output.shape[-1]),
            dtype=encoder_output.dtype,
            device=encoder_output.device,
        )
        self.encoder_output_length = torch.zeros(
            (self.batch_size,), dtype=encoder_output_length.dtype, device=encoder_output_length.device
        )

        self.scores_cpu = torch.zeros(
            (self.batch_size, self.max_time * self.max_symbols),
            dtype=encoder_output.dtype,
            device="cpu",
            pin_memory=True,
        )
        self.labels_cpu = torch.zeros(
            (self.batch_size, self.max_time * self.max_symbols), dtype=torch.int64, device="cpu", pin_memory=True
        )
        self.symbols_per_time_step_cpu = torch.zeros(self.max_time, dtype=torch.int64, device="cpu", pin_memory=True)

        with torch.cuda.stream(torch.cuda.Stream()), torch.inference_mode():
            cu_call(
                cudart.cudaStreamBeginCapture(
                    torch.cuda.current_stream().cuda_stream,
                    cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal,
                )
            )

            self.f = torch.zeros(
                (self.batch_size, 1, self.encoder_output.shape[-1]),
                dtype=encoder_output.dtype,
                device=encoder_output.device,
            )
            hidden = self.caller.decoder.initialize_state(self.f)
            self.last_label = torch.full(
                [self.batch_size], fill_value=self.caller._SOS, dtype=torch.long, device=encoder_output.device
            )
            self.blank_mask = torch.full(
                [self.batch_size], fill_value=0, dtype=torch.bool, device=encoder_output.device
            )
            self.seq_idx_t = torch.zeros([1], dtype=torch.int64, device=encoder_output.device)

            self.scores = torch.zeros(
                (self.max_time * self.max_symbols, self.batch_size),
                dtype=encoder_output.dtype,
                device=encoder_output.device,
            )
            self.labels = torch.zeros(
                (self.max_time * self.max_symbols, self.batch_size), dtype=torch.int64, device=encoder_output.device
            )
            self.symbols_per_time_step = torch.zeros(self.max_time, dtype=torch.int64, device=encoder_output.device)

            # Get max sequence length
            self.max_out_len_t = self.encoder_output_length.max()

            capture_status, _, graph, _, _ = cu_call(
                cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream().cuda_stream)
            )
            assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

            (for_loop_conditional_handle,) = cu_call(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
            for_loop_kernel = create_outer_for_loop_kernel()
            time_idx_ptr = np.array([self.time_idx_t.data_ptr()], dtype=np.uint64)
            max_out_len_ptr = np.array([self.max_out_len_t.data_ptr()], dtype=np.uint64)
            for_loop_args = np.array(
                [for_loop_conditional_handle.getPtr(), time_idx_ptr.ctypes.data, max_out_len_ptr.ctypes.data],
                dtype=np.uint64,
            )

            with with_conditional_node(for_loop_kernel, for_loop_args, for_loop_conditional_handle):
                torch.index_select(self.encoder_output, 1, self.time_idx_t.unsqueeze(0), out=self.f)

                self.not_all_blank_t.fill_(True)
                self.symbols_added_t.fill_(0)

                self.blank_mask.copy_(self.time_idx_t >= self.encoder_output_length)

                while_loop_kernel = create_while_loop_kernel()
                (while_loop_conditional_handle,) = cu_call(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
                not_blank_ptr = np.array([self.not_all_blank_t.data_ptr()], dtype=np.uint64)
                symbols_added_ptr = np.array([self.symbols_added_t.data_ptr()], dtype=np.uint64)
                max_symbols_ptr = np.array([self.max_symbols_t.data_ptr()], dtype=np.uint64)
                while_loop_args = np.array(
                    [
                        while_loop_conditional_handle.getPtr(),
                        not_blank_ptr.ctypes.data,
                        symbols_added_ptr.ctypes.data,
                        max_symbols_ptr.ctypes.data,
                    ],
                    dtype=np.uint64,
                )
                with with_conditional_node(while_loop_kernel, while_loop_args, while_loop_conditional_handle):
                    g, hidden_prime = self.caller._pred_step(
                        self.last_label.unsqueeze(1), hidden, batch_size=self.batch_size
                    )
                    logp = self.caller._joint_step(self.f, g, log_normalize=None)[:, 0, 0, :]

                    v, k = logp.max(1)

                    # Commented out code unnecessarily causes D2H copy, which is synchronous. See pytorch issue #105641
                    # self.scores[self.seq_idx_t, :] = v
                    # self.labels[self.seq_idx_t, :] = k
                    self.scores.index_copy_(0, self.seq_idx_t, v.unsqueeze(0))
                    self.labels.index_copy_(0, self.seq_idx_t, k.unsqueeze(0))

                    self.blank_mask.bitwise_or_(k == self.caller._blank_index)

                    hidden_prime = self.caller.decoder.batch_copy_states_mask(hidden_prime, hidden, self.blank_mask)
                    torch.where(self.blank_mask, self.last_label, k, out=k)
                    self.last_label.copy_(k)

                    hidden[0].copy_(hidden_prime[0])
                    hidden[1].copy_(hidden_prime[1])

                    self.not_all_blank_t.copy_(~self.blank_mask.all())
                    self.symbols_added_t += 1
                    self.seq_idx_t += 1

                self.symbols_per_time_step.index_copy_(0, self.time_idx_t, self.symbols_added_t)
                self.time_idx_t += 1

            self.scores_cpu.copy_(self.scores.transpose(0, 1).contiguous(), non_blocking=True)
            self.labels_cpu.copy_(self.labels.transpose(0, 1).contiguous(), non_blocking=True)
            self.symbols_per_time_step_cpu.copy_(self.symbols_per_time_step, non_blocking=True)

            self.last_label.fill_(self.caller._SOS)
            self.time_idx_t.fill_(0)

            (self.graph,) = cu_call(cudart.cudaStreamEndCapture(torch.cuda.current_stream().cuda_stream))
        (self.graph_exec,) = cu_call(cudart.cudaGraphInstantiate(self.graph, 0))

    def __call__(
        self,
        caller,
        x: torch.Tensor,
        out_len: torch.Tensor,
        device: torch.device,
        partial_hypotheses: Optional[List[rnnt_utils.Hypothesis]] = None,
    ):

        # Need to copy x and out_len into "staging buffers", or do a graph update.

        if partial_hypotheses is not None:
            raise NotImplementedError("`partial_hypotheses` support is not supported")

        assert not caller.preserve_alignments
        assert not caller.preserve_frame_confidence

        batch_size = x.shape[0]
        # ideally we would use out_len.max() here...
        max_time = x.shape[1]

        if torch.is_autocast_enabled():
            x = x.to(torch.get_autocast_gpu_dtype())

        # What do we do if batch_size is actually smaller for the
        # input? Is this a problem? the clone() call will fail...
        if max_time > self.max_time or batch_size > self.batch_size:
            self._reinitialize(max_time, batch_size, x, out_len)

        torch.cuda.nvtx.range_push("Graph exec")
        self.encoder_output[: x.shape[0], : x.shape[1], ...].copy_(x)
        self.encoder_output_length[: out_len.shape[0]].copy_(out_len)
        cu_call(cudart.cudaGraphLaunch(self.graph_exec, torch.cuda.current_stream().cuda_stream))
        cu_call(cudart.cudaStreamSynchronize(torch.cuda.current_stream().cuda_stream))
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("Copy data out")

        torch.cuda.nvtx.range_push("GPU->CPU out_len copy")
        out_len_cpu = out_len.cpu()
        torch.cuda.nvtx.range_pop()

        hypotheses = [
            rnnt_utils.Hypothesis(score=0.0, y_sequence=[], timestep=[], dec_state=None) for _ in range(batch_size)
        ]

        # At batch size 16, this for loop takes 25-31 milliseconds... How can we speed it up?
        # It is likely that most of the overhead is in memory loads...
        # Let's profile just this block
        for i in range(batch_size):
            j = 0
            for t in range(out_len_cpu[i]):
                max_non_blank_symbols = self.symbols_per_time_step_cpu[t]
                for counter in range(max_non_blank_symbols):
                    if self.labels_cpu[i, j] == caller._blank_index:
                        j += max_non_blank_symbols - counter
                        break
                    hypotheses[i].y_sequence.append(self.labels_cpu[i, j])
                    hypotheses[i].timestep.append(t)
                    hypotheses[i].score += self.scores_cpu[i, j]
                    j += 1

        torch.cuda.nvtx.range_pop()

        return hypotheses


