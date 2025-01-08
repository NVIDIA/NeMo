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

from dataclasses import dataclass

from nemo.collections.llm.fn.activation import openai_gelu, quick_gelu

from nemo.collections.vlm.vision.module import CLIPViTConfig


@dataclass
class CLIPViTL_14_336_Config(CLIPViTConfig):
    """Clip vit large patch14 config"""

    vision_model_type: str = "clip"
    patch_dim: int = 14
    img_h: int = 336
    img_w: int = 336
    num_layers: int = 24
    num_attention_heads: int = 16
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    hidden_size: int = 1024
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    ffn_hidden_size: int = 4096
    gated_linear_unit: bool = False
    activation_func: callable = quick_gelu
    kv_channels: int = 64
    num_query_groups: int = 16
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization: str = 'LayerNorm'
    apply_rope_fusion: bool = False

@dataclass
class SigLIPViT400M_14_384_Config(CLIPViTConfig):
    """Siglip so400m patch14 384 config"""

    vision_model_type: str = "siglip"
    patch_dim: int = 14
    img_h: int = 384
    img_w: int = 384
    num_layers: int = 27
    num_attention_heads: int = 16
    add_bias_linear: bool = True
    add_qkv_bias: bool = True
    hidden_size: int = 1152
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    ffn_hidden_size: int = 4304
    gated_linear_unit: bool = False
    activation_func: callable = openai_gelu
    kv_channels: int = 72
    num_query_groups: int = 16
    layernorm_zero_centered_gamma: bool = False
    apply_query_key_layer_scaling: bool = False
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    attention_softmax_in_fp32: bool = True
    normalization: str = 'LayerNorm'
    apply_rope_fusion: bool = False
    qk_layernorm: bool = False
    layernorm_epsilon: float = 1e-6