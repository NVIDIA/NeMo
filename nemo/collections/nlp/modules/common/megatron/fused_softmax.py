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

import torch

try:
    from apex._autocast_utils import _cast_if_autocast_enabled

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


class ScaledMaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, mask, scale):
        import scaled_masked_softmax_cuda_new

        scale_t = torch.tensor([scale])
        softmax_results = scaled_masked_softmax_cuda_new.forward(inputs, mask, scale_t[0])
        ctx.save_for_backward(softmax_results, scale_t)
        return softmax_results

    @staticmethod
    def backward(ctx, output_grads):
        import scaled_masked_softmax_cuda_new

        softmax_results, scale_t = ctx.saved_tensors

        input_grads = scaled_masked_softmax_cuda_new.backward(output_grads, softmax_results, scale_t[0])
        return input_grads, None, None


def scaled_masked_softmax(inputs, mask, scale):
    # input is 4D tensor (b, np, sq, sk)
    args = _cast_if_autocast_enabled(inputs, mask, scale)
    with torch.cuda.amp.autocast(enabled=False):
        return ScaledMaskedSoftmax.apply(*args)


class FusedScaleMaskSoftmax(torch.nn.Module):
    """
    Drop-in replacement for apex FusedSacleMaskSoftmax.
    It removes the seq-len limitations compares with apex one.
    It handles the case that all tokens are masked, it returns zeros for attention score.

    fused operation: scaling + mask + softmax

    Arguments:
        input_in_fp16: flag to indicate if input in fp16 data format.
        input_in_bf16: flag to indicate if input in bf16 data format.
        scaled_masked_softmax_fusion: flag to indicate user want to use softmax fusion
        mask_func: mask function to be applied.
        softmax_in_fp32: if true, softmax in performed at fp32 precision.
        scale: scaling factor used in input tensor scaling.
    """

    def __init__(
        self, input_in_fp16, input_in_bf16, scaled_masked_softmax_fusion, mask_func, softmax_in_fp32, scale,
    ):
        super().__init__()
        if torch.cuda.is_available():
            # this triggers kernel compiling
            import nemo.collections.nlp.modules.common.megatron.fused_kernels
        self.input_in_fp16 = input_in_fp16
        self.input_in_bf16 = input_in_bf16
        if self.input_in_fp16 and self.input_in_bf16:
            raise RuntimeError("both fp16 and bf16 flags cannot be active at the same time.")
        self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
        self.mask_func = mask_func
        self.softmax_in_fp32 = softmax_in_fp32
        self.scale = scale
        self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion

        if not (self.scale is None or softmax_in_fp32):
            raise RuntimeError("softmax should be in fp32 when scaled")
        self.fused_softmax_func = scaled_masked_softmax

    def forward(self, input, mask):
        # [b, np, sq, sk]
        assert input.dim() == 4
        if self.is_kernel_available(mask, *input.size()):
            return self.forward_fused_softmax(input, mask)
        else:
            return self.forward_torch_softmax(input, mask)

    def is_kernel_available(self, mask, b, np, sq, sk):
        if self.scaled_masked_softmax_fusion and 0 < sk:  # user want to fuse  # sk must be 1 ~
            return True
        return False

    def forward_fused_softmax(self, input, mask):
        # input.shape = [b, np, sq, sk]
        scale = self.scale if self.scale is not None else 1.0
        return self.fused_softmax_func(input, mask, scale)

    def forward_torch_softmax(self, input, mask):
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
