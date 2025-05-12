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
from megatron.core.utils import deprecate_inference_params, is_fa_min_version
from torch import Tensor

try:
    from flashattn_hopper.flash_attn_interface import _flash_attn_forward

    HAVE_FA3 = True
except ImportError:
    _flash_attn_forward = None

    HAVE_FA3 = False


def chunkify_cu_seqlens(cu_seqlens, cu_seqlens_padded, attention_chunk_size):
    """
    Splits cumulative sequence lengths into chunks based on attention_chunk_size.

    Args:
        cu_seqlens (list[int]): List of cumulative sequence lengths.
        cu_seqlens_padded (list[int]): List of padded cumulative sequence lengths.
        attention_chunk_size (int): The maximum size of each chunk.

    Returns:
        Tuple[list[int], list[int]]: A tuple containing the new chunked cumulative
        sequence lengths and the new chunked padded cumulative sequence lengths.
    """
    new_cu_seqlens = [cu_seqlens[0]]
    new_cu_seqlens_padded = [cu_seqlens_padded[0]]
    for i in range(1, len(cu_seqlens)):
        start = cu_seqlens[i - 1]
        end = cu_seqlens[i]
        start_padded = cu_seqlens_padded[i - 1]
        end_padded = cu_seqlens_padded[i]

        segment_length = end - start
        num_full_chunks = segment_length // attention_chunk_size

        for j in range(1, num_full_chunks + 1):
            new_index = start + j * attention_chunk_size
            new_cu_seqlens.append(new_index)
            new_index_padded = start_padded + j * attention_chunk_size
            new_cu_seqlens_padded.append(new_index_padded)

        if new_cu_seqlens[-1] != end:
            new_cu_seqlens.append(end)
            new_cu_seqlens_padded.append(end_padded)

    return (torch.tensor(new_cu_seqlens).cuda(non_blocking=True),
            torch.tensor(new_cu_seqlens_padded).cuda(non_blocking=True))


def chunkify(x, attention_chunk_size):
    """
    Pads and reshapes a tensor for chunked processing.

    This function takes an input tensor `x` (typically representing query, key, or value
    in attention mechanisms) and pads its sequence dimension (dim 0) to be a multiple
    of `attention_chunk_size`. It then reshapes the tensor so that the sequence dimension
    is split into chunks, and the chunk dimension is combined with the batch dimension.

    Args:
        x (torch.Tensor): Input tensor, expected shape [seq_length, batch_size, ...].
        attention_chunk_size (int): The desired size of chunks along the sequence dimension.

    Returns:
        torch.Tensor: The reshaped tensor with shape
                      [attention_chunk_size, num_chunks * batch_size, ...].
    """
    # Determine original sequence length. Assume SBHD format.
    assert x.shape[1] == 1, \
        (f"When chunked attention is on, `micro_batch_size` needs to be set to 1. "
         f"Current value is {x.shape[1]}")
    seq_length = x.shape[0]
    # Compute new sequence length (pad_seq_len) as the smallest multiple of attention_chunk_size
    pad_seq_len = ((seq_length + attention_chunk_size - 1) // attention_chunk_size) * attention_chunk_size

    # If padding is needed, create a pad tensor with the same type and device as x.
    if pad_seq_len != seq_length:
        pad_size = pad_seq_len - seq_length
        pad_tensor = torch.zeros(pad_size, *x.shape[1:], device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad_tensor], dim=0)

    # Compute the number of chunks (each of length attention_chunk_size)
    num_chunks = pad_seq_len // attention_chunk_size

    # Reshape from:
    #   [seq_length, batch_size, num_heads, head_dim]
    # to:
    #   [num_chunks, attention_chunk_size, batch_size, num_heads, head_dim]
    x = x.reshape(num_chunks, attention_chunk_size, *x.shape[1:])

    # Transpose the first two dimensions so that the tensor becomes:
    #   [attention_chunk_size, num_chunks, batch_size, num_heads, head_dim]
    x = x.transpose(0, 1)

    # Combine (collapse) the num_chunks and batch_size dimensions into one.
    x = x.reshape(x.shape[0], -1, *x.shape[3:]).contiguous()

    return x


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
