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
#
# Copyright 2018-2019, Mingkun Huang
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
        """
        Helper class to launch the CUDA Kernels to compute the Transducer Loss.

        Args:
            minibatch: Int representing the batch size.
            maxT: The maximum possible acoustic sequence length. Represents T in the logprobs tensor.
            maxU: The maximum possible target sequence length. Represents U in the logprobs tensor.
            alphabet_size: The vocabulary dimension V+1 (inclusive of RNNT blank).
            workspace: An allocated chunk of memory that will be sliced off and reshaped into required
                blocks used as working memory.
            blank: Index of the RNNT blank token in the vocabulary. Generally the first or last token in the vocab.
            num_threads: Number of OMP threads to launch.
            stream: Numba Cuda Stream.
        """
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
        """
        Computes the log softmax denominator of the input activation tensor
        and stores the result in denom.

        Args:
            acts: Activation tensor of shape [B, T, U, V+1]. The input must be represented as a flat tensor
                of shape [B * T * U * (V+1)] to allow pointer indexing.
            denom: A zero tensor of same shape as acts.

        Updates:
            This kernel inplace updates the `denom` tensor
        """
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
        Compute both the loss and the gradients.

        Args:
            acts: A flattened tensor of shape [B, T, U, V+1] representing the activation matrix.
            grad: A flattented zero tensor of same shape as acts.
            costs: A zero vector of length B which will be updated inplace with the log probability costs.
            flat_labels: A flattened matrix of labels of shape [B, U]
            label_lengths: A vector of length B that contains the original lengths of the acoustic sequence.
            input_lengths: A vector of length B that contains the original lengths of the target sequence.

        Updates:
            This will launch kernels that will update inline the following variables:
            -   grads: Gradients of the activation matrix wrt the costs vector.
            -   costs: Negative log likelihood of the forward variable.

        Returns:
            An enum that either represents a successful RNNT operation or failure.
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

        # compute negative log likelihood.
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
