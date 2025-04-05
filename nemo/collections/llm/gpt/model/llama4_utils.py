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
from copy import deepcopy
from typing import TYPE_CHECKING, Optional

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from torch import Tensor

try:
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_layer import TransformerLayer as MCoreTransformerLayer

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

if TYPE_CHECKING:
    pass


def get_llama4_layer_spec(config: "Llama4Config") -> ModuleSpec:
    """Get llama4 layer spec"""

    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    # Use decoder_block_spec: set layer_specs as a list of individual layer specs
    llama4_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=HAVE_TE)

    updated_layer_specs = []
    offset = get_transformer_layer_offset(config)
    for idx, layer_spec in enumerate(llama4_layer_spec.layer_specs):
        layer_no = idx + offset
        updated_layer_spec = deepcopy(layer_spec)

        updated_layer_spec.module = Llama4TransformerLayer
        is_nope_layer = config.nope_layer_interval is not None and (layer_no + 1) % config.nope_layer_interval == 0
        updated_layer_spec.params = {
            'use_rope': not is_nope_layer,
        }
        if config.qk_l2_norm and not is_nope_layer:
            # Use QK Norm
            updated_layer_spec.submodules.self_attention.submodules.q_layernorm = L2Norm
            updated_layer_spec.submodules.self_attention.submodules.k_layernorm = L2Norm
        else:
            updated_layer_spec.submodules.self_attention.submodules.q_layernorm = None
            updated_layer_spec.submodules.self_attention.submodules.k_layernorm = None
        updated_layer_specs.append(updated_layer_spec)

    llama4_layer_spec.layer_specs = updated_layer_specs
    return llama4_layer_spec


class Llama4TransformerLayer(MCoreTransformerLayer):
    """Updated Transformer Layer to enable skip rope in some layers"""

    def __init__(self, use_rope=True, *args, **kwargs):
        self.use_rope = use_rope
        super(Llama4TransformerLayer, self).__init__(*args, **kwargs)

    def forward(self, rotary_pos_emb: Optional[Tensor] = None, **kwargs):
        if not self.use_rope:
            rotary_pos_emb = None
        return super().forward(rotary_pos_emb=rotary_pos_emb, **kwargs)


class L2Norm(torch.nn.Module):
    """
    Applies L2 normalization to the input tensor along the last dimension.

    This module normalizes the input tensor such that the mean of the squared values
    along the last dimension is 1 (within a small epsilon for numerical stability).

    Args:
        hidden_size (int): Expected input shape for normalization (not used internally).
        eps (float, optional): A small value added to the denominator for numerical stability.
            Default: 1e-6.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def _norm(self, x):
        """
        Performs the actual L2 normalization.

        Args:
            x (torch.Tensor): The input tensor to normalize.

        Returns:
            torch.Tensor: The L2-normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass of the L2Norm module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: L2-normalized tensor with the same dtype as input.
        """
        return self._norm(x.float()).type_as(x)
