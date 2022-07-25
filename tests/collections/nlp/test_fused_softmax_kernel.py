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

import pytest
import torch


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


def forward_torch_softmax(input, mask, scale):
    input = input * scale
    mask_output = attention_mask_func(input, mask) if mask is not None else input
    probs = torch.nn.Softmax(dim=-1)(mask_output)
    all_k_masked = mask.all(axis=-1)
    zero_attention_mask = (1.0 - all_k_masked.float())[:, :, :, None]
    probs = probs * zero_attention_mask
    return probs


@pytest.mark.run_only_on('GPU')
class TestFusedSoftmaxKernel:
    @classmethod
    def setup_class(cls):
        # this line will trigger building the kernels
        import nemo.collections.nlp.modules.common.megatron.fused_kernels

    @pytest.mark.unit
    def test_forward(self):
        import scaled_masked_softmax_cuda_new

        batch = 2
        attn = 16
        qlen = 2348
        klen = 3123
        scale_t = torch.tensor([1.0])
        for qlen in [2348, 2322, 1234, 1, 2]:
            for klen in [3123, 1234, 2, 4, 8, 3, 1, 5, 10, 11, 13, 128, 256, 1200, 2048, 4096, 7234, 8192, 10232]:
                inputs = torch.normal(0, 2, (batch, attn, qlen, klen), dtype=torch.float16, device='cuda:0')
                masks = torch.randint(0, 2, (batch, 1, qlen, klen), dtype=torch.bool, device='cuda:0')
                softmax_results = scaled_masked_softmax_cuda_new.forward(inputs, masks, scale_t[0].item())
                softmax_results_torch = forward_torch_softmax(inputs, masks, scale_t[0].item())
                error = (softmax_results_torch - softmax_results).abs().max()
                assert error < 1e-3

    @pytest.mark.unit
    def test_backward(self):
        import scaled_masked_softmax_cuda_new

        batch = 2
        attn = 16
        qlen = 2348
        klen = 3123
        scale_t = torch.tensor([1.0])
        for qlen in [2348, 2322, 1234, 1, 2]:
            for klen in [3123, 1234, 2, 4, 8, 3, 1, 5, 10, 11, 13, 128, 256, 1200, 2048, 4096, 7234, 8192, 10232]:
                inputs = torch.normal(0, 2, (batch, attn, qlen, klen), dtype=torch.float16, device='cuda:0')
                backward = torch.rand_like(inputs, dtype=torch.float16, device='cuda:0')
                masks = torch.randint(0, 2, (batch, 1, qlen, klen), dtype=torch.bool, device='cuda:0')
                softmax_results = scaled_masked_softmax_cuda_new.forward(inputs, masks, scale_t[0].item())
                back_grad = scaled_masked_softmax_cuda_new.backward(backward, softmax_results, scale_t[0].item())

                inputs.requires_grad = True
                softmax_results_torch = forward_torch_softmax(inputs, masks, scale_t[0].item())
                softmax_results_torch.backward(backward)
                error = (back_grad - inputs.grad).abs().max()
                assert error < 1e-3

    @pytest.mark.unit
    def test_allmasked(self):
        import scaled_masked_softmax_cuda_new

        batch = 2
        attn = 16
        qlen = 2348
        klen = 3123
        scale_t = torch.tensor([1.0])
        for qlen in [2348, 2322, 1234, 1, 2]:
            for klen in [3123, 1234, 2, 4, 8, 3, 1, 5, 10, 11, 13, 128, 256, 1200, 2048, 4096, 7234, 8192, 10232]:
                inputs = torch.normal(0, 2, (batch, attn, qlen, klen), dtype=torch.float16, device='cuda:0')
                masks = torch.ones((batch, 1, qlen, klen), dtype=torch.bool, device='cuda:0')
                softmax_results = scaled_masked_softmax_cuda_new.forward(inputs, masks, scale_t[0].item())
                softmax_results_torch = forward_torch_softmax(inputs, masks, scale_t[0].item())
                error = (softmax_results_torch - softmax_results).abs().max()
                assert error < 2e-3
