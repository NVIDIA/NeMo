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

from nemo.utils import logging

try:
    from apex.transformer.functional.fused_softmax import FusedScaleMaskSoftmax

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False
    from nemo.collections.nlp.modules.common.megatron.enums import AttnMaskType, LayerType, ModelType

if HAVE_APEX:

    class MatchedScaleMaskSoftmax(FusedScaleMaskSoftmax):
        """
        fused operation: scaling + mask + softmax
        match the behavior of fused softmax and torch softmax.
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

        def forward_torch_softmax(self, input, mask):
            if self.input_in_float16 and self.softmax_in_fp32:
                input = input.float()

            if self.scale is not None:
                input = input * self.scale
            mask_output = self.mask_func(input, mask) if mask is not None else input
            probs = torch.nn.Softmax(dim=-1)(mask_output)
            if mask is not None:
                all_k_masked = mask.all(axis=-1)
                zero_attention_mask = (1.0 - all_k_masked.type(probs.type()))[:, :, :, None]
                probs = probs * zero_attention_mask

            if self.input_in_float16 and self.softmax_in_fp32:
                if self.input_in_fp16:
                    probs = probs.half()
                else:
                    probs = probs.bfloat16()
            return probs


else:

    class MatchedScaleMaskSoftmax(torch.nn.Module):
        """
            fused operation: scaling + mask + softmax

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
            super().__init__()

            self.input_in_fp16 = input_in_fp16
            self.input_in_bf16 = input_in_bf16
            if self.input_in_fp16 and self.input_in_bf16:
                raise RuntimeError("both fp16 and bf16 flags cannot be active at the same time.")
            self.input_in_float16 = self.input_in_fp16 or self.input_in_bf16
            self.attn_mask_type = attn_mask_type
            self.scaled_masked_softmax_fusion = scaled_masked_softmax_fusion
            self.mask_func = mask_func
            self.softmax_in_fp32 = softmax_in_fp32
            self.scale = scale

            if not (self.scale is None or softmax_in_fp32):
                raise RuntimeError("softmax should be in fp32 when scaled")

        def forward(self, input, mask):
            # [b, np, sq, sk]
            assert input.dim() == 4

            # always use torch code path when apex is unavailable
            return self.forward_torch_softmax(input, mask)

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

        @staticmethod
        def get_batch_per_block(sq, sk, b, np):
            import scaled_masked_softmax_cuda

            return scaled_masked_softmax_cuda.get_batch_per_block(sq, sk, b, np)
