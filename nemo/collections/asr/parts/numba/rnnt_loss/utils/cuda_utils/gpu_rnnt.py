# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import multiprocessing
from typing import Optional

import numba
import torch
from numba import cuda

from nemo.collections.asr.parts.numba.rnnt_loss.utils import global_constants
from nemo.collections.asr.parts.numba.rnnt_loss.utils.cuda_utils import gpu_rnnt_kernel, reduce


class GPURNNT:
    def __init__(
        self, minibatch: int, maxT: int, maxU: int, alphabet_size: int, workspace, blank: int, num_threads: int, stream
    ):
        self.minibatch_ = minibatch
        self.maxT_ = maxT
        self.maxU_ = maxU
        self.alphabet_size_ = alphabet_size
        self.gpu_workspace = cuda.as_cuda_array(
            workspace
        )  # a flat vector of floatX numbers that represents allocated memory slices
        self.blank_ = blank
        self.num_threads_ = num_threads
        self.stream_ = stream  # type: cuda.cudadrv.driver.Stream

        if num_threads > 0:
            numba.set_num_threads(min(multiprocessing.cpu_count(), num_threads))
        else:
            self.num_threads_ = numba.get_num_threads()

    def log_softmax(self, acts: torch.Tensor, denom: torch.Tensor):
        # // trans_acts + pred_acts -> log_softmax denominator
        reduce.reduce_max(
            acts,
            denom,
            rows=self.alphabet_size_,
            cols=self.minibatch_ * self.maxT_ * self.maxU_,
            minus=False,
            stream=self.stream_,
        )

        reduce.reduce_exp(
            acts,
            denom,
            rows=self.alphabet_size_,
            cols=self.minibatch_ * self.maxT_ * self.maxU_,
            minus=True,
            stream=self.stream_,
        )

    def compute_cost_and_score(
        self,
        acts: torch.Tensor,
        grads: Optional[torch.Tensor],
        costs: torch.Tensor,
        labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> global_constants.RNNTStatus:
        """

        Args:
            acts:
            grad:
            costs:
            flat_labels:
            label_lengths:
            input_lengths:

        Returns:

        """
        training = grads is not None
        used_offset = 0

        # // denom
        denom = self.gpu_workspace[used_offset : used_offset + self.maxT_ * self.maxU_ * self.minibatch_]
        used_offset += self.maxT_ * self.maxU_ * self.minibatch_

        # // alphas & betas
        alphas = self.gpu_workspace[used_offset : used_offset + self.maxT_ * self.maxU_ * self.minibatch_]
        used_offset += self.maxT_ * self.maxU_ * self.minibatch_
        betas = self.gpu_workspace[used_offset : used_offset + self.maxT_ * self.maxU_ * self.minibatch_]
        used_offset += self.maxT_ * self.maxU_ * self.minibatch_

        # // logllh
        llForward = self.gpu_workspace[used_offset : used_offset + self.minibatch_]
        used_offset += self.minibatch_
        llBackward = self.gpu_workspace[used_offset : used_offset + self.minibatch_]
        used_offset += self.minibatch_

        if training:
            grads *= 0.0  # zero grads

        ######## START EXECUTION ########
        self.log_softmax(acts, denom)

        # Compute alphas
        gpu_rnnt_kernel.compute_alphas_kernel[self.minibatch_, self.maxU_, self.stream_, 0](
            acts,
            denom,
            alphas,
            llForward,
            input_lengths,
            label_lengths,
            labels,
            self.minibatch_,
            self.maxT_,
            self.maxU_,
            self.alphabet_size_,
            self.blank_,
        )

        if training:
            # Compute betas
            gpu_rnnt_kernel.compute_betas_kernel[self.minibatch_, self.maxU_, self.stream_, 0](
                acts,
                denom,
                betas,
                llBackward,
                input_lengths,
                label_lengths,
                labels,
                self.minibatch_,
                self.maxT_,
                self.maxU_,
                self.alphabet_size_,
                self.blank_,
            )

            # Compute gradient
            grad_blocks_per_grid = self.minibatch_ * self.maxT_ * self.maxU_
            grad_threads_per_block = gpu_rnnt_kernel.GPU_RNNT_THREAD_SIZE
            gpu_rnnt_kernel.compute_grad_kernel[grad_blocks_per_grid, grad_threads_per_block, self.stream_, 0](
                grads,
                acts,
                denom,
                alphas,
                betas,
                llForward,
                input_lengths,
                label_lengths,
                labels,
                self.minibatch_,
                self.maxT_,
                self.maxU_,
                self.alphabet_size_,
                self.blank_,
            )

        # // cost
        costs.copy_to_device(llForward, stream=self.stream_)
        self.stream_.synchronize()

        for mb in range(self.minibatch_):
            costs[mb] = -costs[mb]

        return global_constants.RNNTStatus.RNNT_STATUS_SUCCESS

    def cost_and_grad(
        self,
        acts: torch.Tensor,
        grads: torch.Tensor,
        costs: torch.Tensor,
        pad_labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        if (
            acts is None
            or grads is None
            or costs is None
            or pad_labels is None
            or label_lengths is None
            or input_lengths is None
        ):
            return global_constants.RNNTStatus.RNNT_STATUS_INVALID_VALUE

        return self.compute_cost_and_score(acts, grads, costs, pad_labels, label_lengths, input_lengths)

    def score_forward(
        self,
        acts: torch.Tensor,
        costs: torch.Tensor,
        pad_labels: torch.Tensor,
        label_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ):
        if acts is None or costs is None or pad_labels is None or label_lengths is None or input_lengths is None:
            return global_constants.RNNTStatus.RNNT_STATUS_INVALID_VALUE

        return self.compute_cost_and_score(acts, None, costs, pad_labels, label_lengths, input_lengths)
