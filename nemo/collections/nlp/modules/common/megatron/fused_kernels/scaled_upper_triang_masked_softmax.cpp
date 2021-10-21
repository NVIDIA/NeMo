/* coding=utf-8
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_fp16.h>
#include <torch/extension.h>
#include <vector>

namespace multihead_attn {
namespace fused_softmax {
namespace scaled_upper_triang_masked_softmax {

torch::Tensor fwd_cuda(
    torch::Tensor const& input, 
    float scale_factor);

torch::Tensor bwd_cuda(
    torch::Tensor const& output_grads, 
    torch::Tensor const& softmax_results,
    float scale_factor);

torch::Tensor fwd(torch::Tensor const& input, float scale_factor) {
  AT_ASSERTM(input.dim() == 3, "expected 3D tensor");
  AT_ASSERTM((input.scalar_type() == at::ScalarType::Half) ||
	     (input.scalar_type() == at::ScalarType::BFloat16), 
      "Only fp16 and bf16 are supported");

  return fwd_cuda(input, scale_factor);
}

torch::Tensor bwd(
    torch::Tensor const& output_grads, 
    torch::Tensor const& softmax_results,
    float scale_factor) {

  AT_ASSERTM(output_grads.dim() == 3, "expected 3D tensor");
  AT_ASSERTM(softmax_results.dim() == 3, "expected 3D tensor");

  AT_ASSERTM((output_grads.scalar_type() == at::ScalarType::Half) ||
	     (output_grads.scalar_type() == at::ScalarType::BFloat16), 
      "Only fp16 and bf16 are supported");
  AT_ASSERTM((softmax_results.scalar_type() == at::ScalarType::Half) ||
	     (softmax_results.scalar_type() == at::ScalarType::BFloat16), 
      "Only fp16 and bf16 are supported");

  return bwd_cuda(output_grads, softmax_results, scale_factor);
}

} // end namespace scaled_upper_triang_masked_softmax
} // end namespace fused_softmax
} // end namespace multihead_attn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", 
        &multihead_attn::fused_softmax::scaled_upper_triang_masked_softmax::fwd,
	"Self Multihead Attention scaled, time masked softmax -- Forward.");
  m.def("backward", 
        &multihead_attn::fused_softmax::scaled_upper_triang_masked_softmax::bwd,
	"Self Multihead Attention scaled, time masked softmax -- Backward.");
}
