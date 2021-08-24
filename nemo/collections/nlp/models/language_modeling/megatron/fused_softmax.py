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

import torch
from megatron.model.enums import AttnMaskType


class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply upper triangular mask (typically used in gpt models).
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, scale):
        import scaled_upper_triang_masked_softmax_cuda

        scale_t = torch.tensor([scale])

        softmax_results = scaled_upper_triang_masked_softmax_cuda.forward(inputs, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        import scaled_upper_triang_masked_softmax_cuda

        softmax_results, scale_t = ctx.saved_tensors

        input_grads = scaled_upper_triang_masked_softmax_cuda.backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None


class ScaledMaskedSoftmax(torch.autograd.Function):
    """
    Fused operation which performs following three operations in sequence
    1. Scale the tensor.
    2. Apply the mask.
    3. Perform softmax.
    """

    @staticmethod
    def forward(ctx, inputs, mask, scale):
        import scaled_masked_softmax_cuda

        scale_t = torch.tensor([scale])

        softmax_results = scaled_masked_softmax_cuda.forward(inputs, mask, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        import scaled_masked_softmax_cuda

        softmax_results, scale_t = ctx.saved_tensors

        input_grads = scaled_masked_softmax_cuda.backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None, None


class FusedScaleMaskSoftmax(torch.nn.Module):
    """
    fused operation: scaling + mask + softmax
    Arguments:
        input_in_fp16: flag to indicate if input in fp16 data format.
        attn_mask_type: attention mask type (pad or causal)
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.

    """

    def __init__(
        self,
        input_in_fp16,
        input_in_bf16,
        attn_mask_type,
        scaled_masked_softmax_fusion,
        mask_func,
        softmax_in_fp32,
        scale,
    ):
        super(FusedScaleMaskSoftmax, self).__init__()
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        assert not (
            self.input_in_fp16 and self.input_in_bf16
        ), 'both fp16 and bf16 flags cannot be active at the same time.'
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.attn_mask_type = attn_mask_type
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale

        assert self.scale is None or softmax_in_fp32, "softmax should be in fp32 when scaled"

    def forward(self, input, mask):
        # [b, np, sq, sk]
        assert input.dim() == 4
        data_size = input.size()
        query_seq_len = data_size[-2]
        key_seq_len = data_size[-1]
        attn_batch_size = data_size[0] * data_size[1]

        # constraints on various tensor dimensions to enable warp based
        # optimization and upper triangular optimization (for causal mask)
        custom_kernel_constraint = (
            key_seq_len > 16 and key_seq_len <= 2048 and query_seq_len % 4 == 0 and attn_batch_size % 4 == 0
        )

        # invoke custom kernel
        if (
            self.input_in_float16
            and mask is not None
            and custom_kernel_constraint
            and self.scaled_masked_softmax_fusion
        ):
            scale = self.scale if self.scale is not None else 1.0

            if self.attn_mask_type == AttnMaskType.causal:
                assert query_seq_len == key_seq_len, "causal mask is only for self attention"
                input = input.view(-1, query_seq_len, key_seq_len)
                probs = ScaledUpperTriangMaskedSoftmax.apply(input, scale)
                probs = probs.view(*data_size)
            else:
                assert self.attn_mask_type == AttnMaskType.padding
                probs = ScaledMaskedSoftmax.apply(input, mask, scale)
        else:
            if self.input_in_float16 and self.softmax_in_fp32:
                input = input.float()

            if self.scale is not None:
                input = input * self.scale
            mask_output = self.mask_func(input, mask) if mask is not None else input
            probs = torch.nn.Softmax(dim=-1)(mask_output)

            if self.input_in_float16 and self.softmax_in_fp32:
                if self.input_in_fp16:
                    probs = probs.half()
                else:
                    probs = probs.bfloat16()

        return probs
