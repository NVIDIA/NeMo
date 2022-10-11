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
    from apex.transformer.enums import AttnMaskType
    from apex.transformer.functional.fused_softmax import FusedScaleMaskSoftmax, generic_scaled_masked_softmax

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


class CombinedScaleMaskSoftmax(FusedScaleMaskSoftmax):
    """
    fused operation: scaling + mask + softmax
    Use fast ScaledMaskedSoftmax if the Q/K shape is compliant. Otherwise use generic Fused kernel.
    If scaled_masked_softmax_fusion is False, use torch to compute softmax.
    This is a workaround for https://github.com/NVIDIA/apex/issues/1493.

    Arguments:
        input_in_fp16: flag to indicate if input in fp16 data format.
        input_in_bf16: flag to indicate if input in bf16 data format.
        attn_mask_type: attention mask type (pad or causal)
        scaled_masked_softmax_fusion: flag to indicate user want to use softmax fusion
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
        super().__init__(
            input_in_fp16,
            input_in_bf16,
            attn_mask_type,
            scaled_masked_softmax_fusion,
            mask_func,
            softmax_in_fp32,
            scale,
        )
        self.generic_scaled_masked_softmax_fusion = generic_scaled_masked_softmax

    def is_generic_kernel_available(self, mask, b, np, sq, sk):
        if self.scaled_masked_softmax_fusion and 0 < sk:  # user want to fuse  # sk must be 1 ~
            return True
        return False

    def forward(self, input, mask):
        # [b, np, sq, sk]
        assert input.dim() == 4

        if self.is_kernel_available(mask, *input.size()):
            return self.forward_fused_softmax(input, mask)
        elif self.is_generic_kernel_available(mask, *input.size()):
            return self.forward_generic_softmax(input, mask)
        else:
            return self.forward_torch_softmax(input, mask)

    def forward_generic_softmax(self, input, mask):
        # input.shape = [b, np, sq, sk]
        scale = self.scale if self.scale is not None else 1.0
        return self.generic_scaled_masked_softmax_fusion(input, mask, scale)
