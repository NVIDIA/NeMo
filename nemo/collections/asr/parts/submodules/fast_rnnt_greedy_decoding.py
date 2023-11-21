from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import nvrtc
from cuda import cuda, cudart

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
    if error != cudart.cudaSuccess:
        raise Exception(f"CUDA failure! {error}")
    else:
        return *others

def create_while_loop_kernel():
    kernel_string = """\
    extern "C" __global__
    void while_loop_conditional(cudaGraphConditionalHandle handle, const bool *not_blank, const long *symbols_added, const long *max_symbols)
    {
     if (*not_blank && *symbols_added < *max_symbols) {
         cudaGraphSetConditional(handle, true);
     } else {
         cudaGraphSetConditional(handle, false);
     }
    }
    """

    err, prog = nvrtc.nvrtcCreateProgram(str.encode(kernel_string), b"while_loop_conditional.cu", 0, [], [])

    ASSERT_DRV(err)

    # Compile program
    opts = [b"--gpu-architecture=compute_80"]
    err, = nvrtc.nvrtcCompileProgram(prog, 2, opts)
    ASSERT_DRV(err)

    # Get PTX from compilation
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    ASSERT_DRV(err)
    ptx = b" " * ptxSize
    err, = nvrtc.nvrtcGetPTX(prog, ptx)
    ASSERT_DRV(err)

    ptx = np.char.array(ptx)
    err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
    ASSERT_DRV(err)
    err, kernel = cuda.cuModuleGetFunction(module, b"while_loop_conditional")
    ASSERT_DRV(err)

    return kernel


class RNNTGreedyDecodeFast:
    def __init__(self, max_symbols: int, cuda_device):
        assert max_symbols is not None

        self.symbols_added_t = torch.tensor(0, dtype=torch.int64, device=cuda_device)
        self.max_symbols_t = torch.tensor(max_symbols, dtype=torch.int64, device=cuda_device)
        self.not_blank_t = torch.tensor(True, dtype=torch.bool, device=cuda_device)

    def __call__(
        self,
        caller,
        x: torch.Tensor,
        out_len: torch.Tensor,
        device: torch.device,
        partial_hypotheses: Optional[List[rnnt_utils.Hypothesis]] = None,
    ):
        if partial_hypotheses is not None:
            raise NotImplementedError("`partial_hypotheses` support is not supported")

        assert not caller.preserve_alignments
        assert not caller.preserve_frame_confidence

        with torch.inference_mode():
            # x: [B, T, D]
            # out_len: [B]
            # device: torch.device

            # Initialize list of Hypothesis
            batchsize = x.shape[0]
            hypotheses = [
                rnnt_utils.Hypothesis(score=0.0, y_sequence=[], timestep=[], dec_state=None) for _ in range(batchsize)
            ]

            # Initialize Hidden state matrix (shared by entire batch)
            # TODO: Need to make an abstract method for initializing states
            hidden = caller.decoder.initialize_state(x)

            # Last Label buffer + Last Label without blank buffer
            # batch level equivalent of the last_label
            self.last_label = torch.full([batchsize, 1], fill_value=caller._SOS, dtype=torch.long, device=device)

            # Mask buffers
            blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool, device=device)
            blank_mask_prev = None

            # We have three conditionals
            # 1) the for loop over the encoder outputs
            # 2) the while loop until all are blank
            #   We would like to copy the greedy outputs from device to host. 
            #   Can't use a cudaEvent in the body of a loop though.

            # Get max sequence length
            max_out_len = out_len.max()
            # This implicitly codes  blocking memory copy from GPU to CPU
            max_out_len = max_out_len.item()

            self.time_idx_t = torch.tensor(0, dtype=torch.int64, device=cuda_device)

            for time_idx in range(max_out_len):
                # This is problematic for cuda graphs...
                self.f.copy_(x.narrow_copy(dim=1, start=time_idx, length=1))  # [B, 1, D]


                if time_idx == 0:
                    self.capture_inner_loop(self.f, time_idx, caller)

                cudart.cudaGraphLaunch(self.graph_exec, torch.cuda.current_stream())

                self.last_label.fill_(caller._SOS)


    def capture_inner_loop(self, f, time_idx, caller):
        cudart.cudaStreamBeginCapture(
            torch.cuda.current_stream(),
            cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal)

        # Prepare t timestamp batch variables
        # This should not do a sync!
        self.not_blank_t.fill_(True)
        self.symbols_added_t.fill_(0)

        # Update blank mask with time mask
        # Batch: [B, T, D], but Bi may have seq len < max(seq_lens_in_batch)
        # Forcibly mask with "blank" tokens, for all sample where current time step T > seq_len
        blank_mask = time_idx >= out_len  # out_len is [B, ]
        # Could do this in a separate stream
        blank_mask_prev = blank_mask.clone()

        g, hidden_prime = caller._pred_step(caller._SOS, hidden, batch_size=batchsize)

        capture_status, *outputs = cu_call(cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream()))
        assert capture_status == cudart.cudaStreamCaptureStatusActive
        _, graph, dependencies = outputs
        while_loop_conditional = cu_call(cudart.cudaGraphConditionalHandleCreate(graph))

        while_loop_kernel = create_while_loop_kernel()
        while_loop_args = np.array([while_loop_conditional.ctypes.data, self.not_blank_t.data(), self.symbols_added_t.data(), self.max_symbols_t.data()], dtype=np.uint64)

        cuda.cuLaunchKernel(
            while_loop_kernel,
            1, 1, 1,
            1, 1, 1,
            0,
            torch.cuda.current_stream(),
            while_loop_args.ctypes.data,
            0
        )

        capture_status, *outputs = cu_call(cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream()))
        assert capture_status == cudart.cudaStreamCaptureStatusActive
        _, graph, dependencies = outputs

        params = cudart.cudaGraphNodeParams()
        params.type = cudart.cudaGraphNodeType.cudaGraphNodeTypeConditional
        params.conditional.handle = while_loop_conditional
        params.conditional.type   = cudart.cudaGraphConditionalNodeType.cudaGraphCondTypeWhile
        params.conditional.size   = 1

        node = cu_call(cudart.cudaGraphAddNode(graph, dependencies, len(dependencies), params))
        body_graph = params.conditional.phGraph_out[0]
        cu_call(cudart.cudaStreamUpdateCaptureDependencies(torch.cuda.current_stream(), [node], 1, cudart.cudaStreamSetCaptureDependencies))
        body_stream = cu_call(cudart.cudaStreamCreate())
        cu_call(cudart.cudaStreamBeginCaptureToGraph(body_stream, body_graph, None, None, 0, cudart.cudaStreamCaptureModeTODO))
        previous_stream = torch.cuda.current_stream()
        torch.cuda.set_stream(body_stream)

        # Need to reset last_label to _SOS at every outer iteration
        g, hidden_prime = caller._pred_step(self.last_label, hidden, batch_size=batchsize)

        logp = caller._joint_step(f, g, log_normalize=None)[
            :, 0, 0, :
        ]

        # Get index k, of max prob for batch
        v, k = logp.max(1)

        # Update blank mask with current predicted blanks
        # This is accumulating blanks over all time steps T and all target steps min(max_symbols, U)
        blank_mask.bitwise_or_(k == caller._blank_index)
        blank_mask_prev.bitwise_or_(blank_mask)

        all_blank_t = blank_mask.all()

        self.not_blank_t = not all_blank_t
        # There are some non-blank outputs. We need to keep going then.
        # Recover prior state for all samples which predicted blank now/past
        hidden_prime = caller.decoder.batch_copy_states_mask(
            hidden_prime, hidden, blank_mask)

        # Recover prior predicted label for all samples which predicted blank now/past
        # Basically, set k to blank if blank was predicted before?
        k[blank_mask] = self.last_label[blank_mask, 0]

        # Update new label and hidden state for next iteration
        # last_label
        self.last_label.copy_(k.view(-1, 1))
        # TODO: Make this swap happy for cuda graphs.
        hidden = hidden_prime

        for kidx, ki in enumerate(k):
            if blank_mask[kidx] == 0: # GPU -> CPU copy
                hypotheses[kidx].y_sequence.append(ki)
                hypotheses[kidx].timestep.append(time_idx)
                hypotheses[kidx].score += float(v[kidx])

        self.symbols_added += 1

        torch.cuda.set_stream(previous_stream)
        cu_call(cudart.cudaStreamDestroy(body_stream))

        self.time_idx_t += 1

        # Preserve states
        for batch_idx in range(batchsize):
            hypotheses[batch_idx].dec_state = caller.decoder.batch_select_state(hidden, batch_idx)

        return hypotheses
        
