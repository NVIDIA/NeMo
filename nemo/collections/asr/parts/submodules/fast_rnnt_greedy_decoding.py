import contextlib
import ctypes
from dataclasses import dataclass, field
from itertools import product
import time
from typing import List, Optional, Tuple, Union

from cuda import cuda, cudart, nvrtc

import numpy as np
import torch
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
    opts = [b"--gpu-architecture=compute_80", b"--include-path=/home/dgalvez/scratch/cuda/cuda-12.3/include/"]
    err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    err, size = nvrtc.nvrtcGetProgramLogSize(prog)
    buf = b" " * size
    err, = nvrtc.nvrtcGetProgramLog(prog, buf)
    print(buf.decode("utf-8"))
    ASSERT_DRV(err)

    # Get PTX from compilation
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    ASSERT_DRV(err)
    ptx = b" " * ptxSize
    err, = nvrtc.nvrtcGetPTX(prog, ptx)
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
    kernel_string = """\
    #include "/home/dgalvez/scratch/cuda/cuda-12.3/include/cuda_device_runtime_api.h"
    #include "/home/dgalvez/scratch/cuda/cuda-12.3/include/driver_types.h"

    // TODO: Figure out why the includes don't work.
    typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;

    extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);

    extern "C" __global__
    void for_loop_conditional(cudaGraphConditionalHandle handle, const long *time_idx, const long *trip_count)
    {
     // printf("Time idx: %ld, trip count: %ld\\n", *time_idx, *trip_count);
     // we have problems ending only when this is set to true...
     cudaGraphSetConditional(handle, *time_idx < *trip_count);
     // static int i = 0;
     // cudaGraphSetConditional(handle, ++i < 10); // *time_idx < *trip_count);
    }
    """
    return run_nvrtc(kernel_string, b"for_loop_conditional")

# Observations: If cudaGraphSetConditional is true once, the kernel never completes...
# The GPU is doing *something*. I'm just not sure what...
# No way to query that at runtime...

def create_while_loop_kernel():
    kernel_string = """\
    // TODO: Figure out why the includes don't work.
    typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;

    extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);

    extern "C" __global__
    void while_loop_conditional(cudaGraphConditionalHandle handle, const bool *not_blank, const long *symbols_added, const long *max_symbols)
    {
     // printf("symbols_added: %ld, not_blank: %d, max_symbols:%ld\\n", *symbols_added, *not_blank, *max_symbols);
     cudaGraphSetConditional(handle, *not_blank && *symbols_added < *max_symbols);
    }
    """
    return run_nvrtc(kernel_string, b"while_loop_conditional")


@contextlib.contextmanager
def with_conditional_node(while_loop_kernel, while_loop_args, while_loop_conditional_handle):
    capture_status, _, graph, _, _ = cu_call(cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream().cuda_stream))
    assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

    cuda.cuLaunchKernel(
        while_loop_kernel,
        1, 1, 1,
        1, 1, 1,
        0,
        torch.cuda.current_stream().cuda_stream,
        while_loop_args.ctypes.data,
        0
    )

    capture_status, _, graph, dependencies, _  = cu_call(cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream().cuda_stream))
    assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

    # import ipdb; ipdb.set_trace()
    params = cudart.cudaGraphNodeParams()
    params.type = cudart.cudaGraphNodeType.cudaGraphNodeTypeConditional
    params.conditional.handle = while_loop_conditional_handle
    params.conditional.type   = cudart.cudaGraphConditionalNodeType.cudaGraphCondTypeWhile
    params.conditional.size   = 1
    # params.conditional.phGraph_out  =  [None]
    # import ipdb; ipdb.set_trace()

    driver_params = cuda.CUgraphNodeParams()
    driver_params.type = cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL
    driver_params.conditional.handle = while_loop_conditional_handle
    driver_params.conditional.type   = cuda.CUgraphConditionalNodeType.CU_GRAPH_COND_TYPE_WHILE
    driver_params.conditional.size   = 1
    # Work around until https://github.com/NVIDIA/cuda-python/issues/55 is fixed
    driver_params.conditional.phGraph_out  = [None]
    ctx, = cu_call(cuda.cuCtxGetCurrent())
    driver_params.conditional.ctx    = ctx

    # Use driver API here because of bug in cuda-python runtime API: https://github.com/NVIDIA/cuda-python/issues/55
    # node, = cu_call(cudart.cudaGraphAddNode(graph, dependencies, len(dependencies), driver_params))
    node, = cu_call(cuda.cuGraphAddNode(graph, dependencies, len(dependencies), driver_params))
    body_graph = driver_params.conditional.phGraph_out[0]

    cu_call(cudart.cudaStreamUpdateCaptureDependencies(torch.cuda.current_stream().cuda_stream, [node], 1, cudart.cudaStreamUpdateCaptureDependenciesFlags.cudaStreamSetCaptureDependencies))
    body_stream = torch.cuda.Stream()
    previous_stream = torch.cuda.current_stream()
    cu_call(cudart.cudaStreamBeginCaptureToGraph(body_stream.cuda_stream, body_graph, None, None, 0, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal))
    torch.cuda.set_stream(body_stream)

    yield body_stream, body_graph

    cuda.cuLaunchKernel(
        while_loop_kernel,
        1, 1, 1,
        1, 1, 1,
        0,
        body_stream.cuda_stream,
        while_loop_args.ctypes.data,
        0
    )

    cudart.cudaStreamEndCapture(body_stream.cuda_stream)

    torch.cuda.set_stream(previous_stream)


class RNNTGreedyDecodeFast:
    def __init__(self, max_symbols: int, cuda_device):
        assert max_symbols is not None

        self.symbols_added_t = torch.tensor(0, dtype=torch.int64, device=cuda_device)
        self.max_symbols_t = torch.tensor(max_symbols, dtype=torch.int64, device=cuda_device)
        self.max_symbols = max_symbols
        self.not_blank_t = torch.tensor(True, dtype=torch.bool, device=cuda_device)

        self.cuda_device = cuda_device

        self.time_idx_t = torch.tensor(0, dtype=torch.int64, device=self.cuda_device)
        self.max_out_len = torch.tensor(0, dtype=torch.int64, device=self.cuda_device)
        # TODO: infer dtype correctly
        # self.f = torch.zeros((batch_size, TODO), dtype=torch.float32, device=self.cuda_device)


    def __call__(
        self,
        caller,
        x: torch.Tensor,
        out_len: torch.Tensor,
        device: torch.device,
        partial_hypotheses: Optional[List[rnnt_utils.Hypothesis]] = None,
    ):

        print("GALVEZ:", type(self), type(caller))
        if partial_hypotheses is not None:
            raise NotImplementedError("`partial_hypotheses` support is not supported")

        assert not caller.preserve_alignments
        assert not caller.preserve_frame_confidence

        batch_size = x.shape[0]
        max_time = x.shape[1]
        
        self.scores_cpu = torch.zeros((max_time * self.max_symbols, batch_size), dtype=x.dtype, device="cpu", pin_memory=True)
        self.labels_cpu = torch.zeros((max_time * self.max_symbols, batch_size), dtype=torch.int64, device="cpu", pin_memory=True)
        self.symbols_per_time_step_cpu = torch.zeros(max_time, dtype=torch.int64, device="cpu", pin_memory=True)

        with torch.cuda.stream(torch.cuda.Stream()), torch.inference_mode():
            cu_call(cudart.cudaStreamBeginCapture(torch.cuda.current_stream().cuda_stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal))
            # x: [B, T, D]
            # out_len: [B]
            # device: torch.device

            # Initialize list of Hypothesis
            hypotheses = [
                rnnt_utils.Hypothesis(score=0.0, y_sequence=[], timestep=[], dec_state=None) for _ in range(batch_size)
            ]

            vocab_size = caller.num_tokens

            # Initialize Hidden state matrix (shared by entire batch)
            # TODO: Need to make an abstract method for initializing states
            hidden = caller.decoder.initialize_state(x)

            # Last Label buffer + Last Label without blank buffer
            # batch level equivalent of the last_label
            self.last_label = torch.full([batch_size], fill_value=caller._SOS, dtype=torch.long, device=device)

            # Mask buffers
            self.blank_mask = torch.full([batch_size], fill_value=0, dtype=torch.bool, device=device)

            self.seq_idx_t = torch.zeros([1], dtype=torch.int64, device=device)

            self.f = torch.zeros((batch_size, 1, x.shape[-1]), dtype=x.dtype, device=device)

            self.scores = torch.zeros((max_time * self.max_symbols, batch_size), dtype=x.dtype, device=device)
            self.labels = torch.zeros((max_time * self.max_symbols, batch_size), dtype=torch.int64, device=device)
            self.symbols_per_time_step = torch.zeros(max_time, dtype=torch.int64, device=device)

            # We have three conditionals
            # 1) the for loop over the encoder outputs
            # 2) the while loop until all are blank
            #   We would like to copy the greedy outputs from device to host. 
            #   Can't use a cudaEvent in the body of a loop though.

            # Get max sequence length
            self.max_out_len = out_len.max()

            capture_status, _, graph, _, _ = cu_call(cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream().cuda_stream))
            assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

            for_loop_conditional_handle, = cu_call(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
            for_loop_kernel = create_outer_for_loop_kernel()
            time_idx_ptr = np.array([self.time_idx_t.data_ptr()], dtype=np.uint64)
            max_out_len_ptr = np.array([self.max_out_len.data_ptr()], dtype=np.uint64)
            for_loop_args = np.array([for_loop_conditional_handle.getPtr(), time_idx_ptr.ctypes.data, max_out_len_ptr.ctypes.data], dtype=np.uint64)

            self.time_idx_t.fill_(0)

            with with_conditional_node(for_loop_kernel, for_loop_args, for_loop_conditional_handle):
                torch.index_select(x, 1, self.time_idx_t.unsqueeze(0), out=self.f)

                self.not_blank_t.fill_(True)
                self.symbols_added_t.fill_(0)

                self.blank_mask.copy_(self.time_idx_t >= out_len)
                
                while_loop_kernel = create_while_loop_kernel()
                while_loop_conditional_handle, = cu_call(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
                not_blank_ptr = np.array([self.not_blank_t.data_ptr()], dtype=np.uint64)
                symbols_added_ptr = np.array([self.symbols_added_t.data_ptr()], dtype=np.uint64)
                max_symbols_ptr = np.array([self.max_symbols_t.data_ptr()], dtype=np.uint64)
                while_loop_args = np.array([while_loop_conditional_handle.getPtr(),
                                            not_blank_ptr.ctypes.data,
                                            symbols_added_ptr.ctypes.data,
                                            max_symbols_ptr.ctypes.data],
                                           dtype=np.uint64)
                with with_conditional_node(while_loop_kernel, while_loop_args, while_loop_conditional_handle):
                    g, hidden_prime = caller._pred_step(self.last_label.unsqueeze(1), hidden, batch_size=batch_size)
                    logp = caller._joint_step(self.f, g, log_normalize=None)[
                        :, 0, 0, :
                    ]

                    # torch.max(logp, 1, out=(self.scores[self.seq_idx_t, :], self.labels[self.seq_idx_t, :]))
                    # v = self.scores[self.seq_idx_t, :]
                    # k = self.labels[self.seq_idx_t, :]
                    v, k = logp.max(1)

                    self.scores.index_copy_(0, self.seq_idx_t, v.unsqueeze(0))
                    self.labels.index_copy_(0, self.seq_idx_t, k.unsqueeze(0))
                    # Causes D2H copy. See pytorch issue #105641
                    # self.scores[self.seq_idx_t, :] = v
                    # self.labels[self.seq_idx_t, :] = k

                    self.blank_mask.bitwise_or_(k == caller._blank_index)

                    # You want to recover the prior state for anything
                    # that has predicted blank. Why? Oh, to retain it
                    # for the next outer loop iteration.
                    hidden_prime = caller.decoder.batch_copy_states_mask(hidden_prime, hidden, self.blank_mask)

                    # This seems wrong. Do I need to negate this?
                    k.masked_scatter_(self.blank_mask, self.last_label)
                    # This doesn't seem right. Why is my last label blank? It should be SOS, right?
                    # I should not copy k if last_label is SOS, right?
                    self.last_label.copy_(k)

                    # It seems that I am unconditionally copying. That is wrong... I should do a masked copy
                    hidden[0].copy_(hidden_prime[0])
                    hidden[1].copy_(hidden_prime[1])

                    self.not_blank_t = ~self.blank_mask.all()
                    self.symbols_added_t += 1
                    self.seq_idx_t += 1

                self.symbols_per_time_step.index_copy_(0, self.time_idx_t, self.symbols_added_t)
                # self.symbols_per_time_step[self.time_idx_t] = self.symbols_added_t
                self.time_idx_t += 1

            self.scores_cpu.copy_(self.scores, non_blocking=True)
            self.labels_cpu.copy_(self.labels, non_blocking=True)
            self.symbols_per_time_step_cpu.copy_(self.symbols_per_time_step, non_blocking=True)

            self.last_label.fill_(caller._SOS)
            self.time_idx_t.fill_(0)

            self.graph, = cu_call(cudart.cudaStreamEndCapture(torch.cuda.current_stream().cuda_stream))
        # Could unindent here, but we want to keep torch.inference_mode,bleh
        self.graph_exec, = cu_call(cudart.cudaGraphInstantiate(self.graph, 0))
        cudart.cudaGraphDebugDotPrint(self.graph, b"graph2.dot", cudart.cudaGraphDebugDotFlags.cudaGraphDebugDotFlagsVerbose)

        with torch.inference_mode():
            start = time.time()
            torch.cuda.cudart().cudaProfilerStart()
            cu_call(cudart.cudaGraphLaunch(self.graph_exec, torch.cuda.current_stream().cuda_stream))
            cu_call(cudart.cudaStreamSynchronize(torch.cuda.current_stream().cuda_stream))
            end = time.time()
            print("total time:", end - start)

            torch.set_printoptions(threshold=100_000)
            print("GALVEZ:", self.symbols_per_time_step_cpu)
            print("GALVEZ:scores=", self.scores_cpu)
            print("GALVEZ:labels=", self.labels_cpu)
            print("GALVEZ:symbols_per_time_step=", self.symbols_per_time_step_cpu)


            torch.cuda.nvtx.range_push("Copy data out")
            # js = torch.zeros(batch_size, dtype=torch.int64, device="cpu")
            j = 0
            for t in range(max_time):
                max_non_blank_symbols = self.symbols_per_time_step_cpu[t]
                print("GALVEZ:", t, max_non_blank_symbols)
                for _ in range(max_non_blank_symbols):
                    for i in range(batch_size):
                        if self.labels_cpu[j, i] == caller._blank_index:
                            # Ooops! This is not correct!!!!! It's continue... It's fine...
                            continue
                        hypotheses[i].y_sequence.append(self.labels_cpu[j, i])
                        hypotheses[i].timestep.append(t)
                        hypotheses[i].score += self.scores_cpu[j, i]
                    j += 1
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()

            print("NEW:", hypotheses)

            import ipdb; ipdb.set_trace()

            return hypotheses
