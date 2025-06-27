# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from functools import partial

import torch

from megatron.core.transformer.transformer_layer import TransformerLayer


def _bias_dropout_add_func_layer_scaling(ls, x_with_bias, residual, prob, training):
    """Handle InternViT's layer scaling."""
    x, bias = x_with_bias  # unpack
    residual = residual if residual.dtype == x.dtype else residual.to(x.dtype)
    if bias is not None:
        x = x + bias
        out = torch.nn.functional.dropout(x, p=prob, training=training)
        out = residual + out * ls
        return out
    else:
        out = torch.nn.functional.dropout(x, p=prob, training=training)
        out = residual + out * ls
        return out


def bias_dropout_add_unfused_layer_scaling(ls, training):
    """Bias-dropout-add as in Megatron but with added LayerScaling handling."""

    def _bias_dropout_add(x_with_bias, residual, prob):
        #
        return _bias_dropout_add_func_layer_scaling(ls, x_with_bias, residual, prob, training)

    return _bias_dropout_add


def get_bias_dropout_add_layer_scaling(ls, training, fused):
    """Bias-dropout-add as in Megatron but with added LayerScaling handling."""
    assert not fused, "Fused bias-dropout-add not implemented for InternViT."
    return bias_dropout_add_unfused_layer_scaling(ls, training)


class LayerScalingTransformerLayer(TransformerLayer):
    """Add InternViT specialties to our default TransformerLayer."""

    def __init__(self, *args, **kwargs):
        # pylint: disable=C0115,C0116
        super().__init__(*args, **kwargs)
        self.ls1 = torch.nn.Parameter(torch.ones(self.config.hidden_size))
        self.ls2 = torch.nn.Parameter(torch.ones(self.config.hidden_size))

        self.self_attn_bda = partial(self.self_attn_bda, self.ls1)
        self.mlp_bda = partial(self.mlp_bda, self.ls2)
