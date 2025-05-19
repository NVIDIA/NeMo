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
from typing import Optional, Tuple, Union

import torch
from einops import rearrange
from megatron.core import parallel_state
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.attention import SelfAttention as MCoreSelfAttention
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import L2Norm
from megatron.core.utils import deprecate_inference_params
from torch import Tensor

try:
    from nvidia_chunked_flash_attn.flash_attn_interface import (
        flash_attn_varlen_func as flash_decode_and_prefill_kernel,
    )
except ImportError:
    flash_decode_and_prefill_kernel = None


def get_llama4_layer_spec(config) -> ModuleSpec:
    """Get llama4 layer spec"""

    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    # Use decoder_block_spec: set layer_specs as a list of individual layer specs
    llama4_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True)

    updated_layer_specs = []
    offset = get_transformer_layer_offset(config)
    for idx, layer_spec in enumerate(llama4_layer_spec.layer_specs):
        layer_no = idx + offset
        updated_layer_spec = deepcopy(layer_spec)
        is_nope_layer = config.no_rope_freq is not None and config.no_rope_freq[layer_no]
        updated_layer_spec.submodules.self_attention.module = MCoreSelfAttention
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
