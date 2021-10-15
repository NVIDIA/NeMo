# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from nemo.collections.nlp.modules.common.megatron.utils import AutocastModuleWrapper


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


@torch.jit.script
def bias_dropout_add_fused_train_(
    x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, prob: float
) -> torch.Tensor:
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, True)


class BiasDropoutAddFusedTrain(AutocastModuleWrapper):
    def __init__(self, fp16=False, bf16=False):
        super(BiasDropoutAddFusedTrain, self).__init__(fp16, bf16)

        self.func = bias_dropout_add_fused_train_

    def forward(self, x, bias, residual, prob):
        return self.autocast_forward(x, bias, residual, prob)


@torch.jit.script
def bias_dropout_add_fused_inference_(
    x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, prob: float
) -> torch.Tensor:
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, False)


class BiasDropoutAddFusedInference(AutocastModuleWrapper):
    def __init__(self, fp16=False, bf16=False):
        super(BiasDropoutAddFusedInference, self).__init__(fp16, bf16)

        self.func = bias_dropout_add_fused_inference_

    def forward(self, x, bias, residual, prob):
        return self.autocast_forward(x, bias, residual, prob)
