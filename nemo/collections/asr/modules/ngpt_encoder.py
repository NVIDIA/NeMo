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
from rotary_embedding_torch import RotaryEmbedding

import math
from collections import OrderedDict
from dataclasses import dataclass
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention.flex_attention import create_block_mask

import torch
import torch.distributed
from omegaconf import DictConfig, ListConfig, open_dict
import torch.nn as nn
import random
import pdb
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AccessMixin
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType, SpectrogramType
from nemo.utils import logging
from nemo.collections.asr.models.configs import CacheAwareStreamingConfig
from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder

try:
    from flash_attn import flash_attn_func
except ImportError:

    def flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=1.0,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
    ):
        """Quick and dirty implementation for prototyping."""
        return nn.functional.softmax(q @ (k * softmax_scale).transpose(2, 3), dim=-1) @ v


__all__ = ['NGPTEncoder']


class NGPTEncoder(NeuralModule,StreamingEncoder, Exportable, AccessMixin):
    """
    Transformer encoder based on nGPT for ASR.
    Based on this paper:
    'nGPT: Normalized Transformer with Representation Learning on the Hypersphere' by Ilya Loshchilov et al.
    https://github.com/NVIDIA/ngpt
    """

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        dev = next(self.parameters()).device
        input_example = torch.randn(max_batch, self._feat_in, max_dim, device=dev)
        input_example_length = torch.randint(max_dim // 4, max_dim, (max_batch,), device=dev, dtype=torch.int64)
        all_input_example = tuple([input_example, input_example_length])
        return all_input_example

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def input_types_for_export(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def output_types_for_export(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            }
        )

    @property
    def disabled_deployment_input_names(self):
        return set()

    @property
    def disabled_deployment_output_names(self):
        return set()

    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        base_scale: float = 1 / (1024**0.5),  # 1/sqrt(d_model)
        n_heads=4,
        causal_downsampling=False,
        subsampling='striding',
        subsampling_factor=4,
        subsampling_conv_chunking_factor=1,
        subsampling_conv_channels=256,
        att_context_size=None,
        att_context_probs=None,
        att_context_style='regular',
        use_bias=False,
        dropout=0.1,
        use_nGPT=True,
    ):
        super().__init__()
        # adding all necessary parameters for caching
        (
            self.att_context_size_all,
            self.att_context_size,
            self.att_context_probs,
            self.conv_context_size,
        ) = self._calc_context_sizes(
            att_context_style=att_context_style,
            att_context_size=att_context_size,
            att_context_probs=att_context_probs,
        )
        self.att_context_style = att_context_style
        self.subsampling_factor = subsampling_factor
        self._feat_out = d_model
        self._feat_in = feat_in
        self.layers = n_layers
        self.d_model = d_model
        if subsampling == "ngpt-frame-stack":
            self.pre_encode = NGPTStackingSubsampling(
                subsampling_factor=subsampling_factor,
                feat_in=feat_in,
                feat_out=d_model,
                use_bias=use_bias,
                base_scale=base_scale,
            )
        self.ngpt = GPT(
            config=GPTConfig(
                n_layer=n_layers,
                n_head=n_heads,
                n_embd=d_model,
                base_scale=base_scale,
                use_nGPT=use_nGPT,
                dropout=dropout,
                bias=use_bias,
                att_context_size=self.att_context_size,
            )
        )
        self.setup_streaming_params()

    def get_initial_cache_state(self, batch_size=1, dtype=torch.float32, device=None, max_dim=0):

        if device is None:
            device = next(self.parameters()).device
        if max_dim > 0:
            create_tensor = torch.randn
        else:
            create_tensor = torch.zeros
        
        cache_last_channel = create_tensor(
            (
                self.layers,
                batch_size,
                self.streaming_cfg.last_channel_cache_size,
                self.d_model,
            ),
            device=device,
            dtype=dtype,
        )

        if max_dim > 0:
            cache_last_channel_len = torch.randint(
                0,
                min(max_dim, self.streaming_cfg.last_channel_cache_size),
                (batch_size,),
                device=device,
                dtype=torch.int64,
            )
            for i in range(batch_size):
                cache_last_channel[:, i, cache_last_channel_len[i] :, :] = 0    
        else:
            cache_last_channel_len = torch.zeros(batch_size, device=device, dtype=torch.int64)

        return cache_last_channel, None, cache_last_channel_len #Returning None for convolution cache support
    

    def set_default_att_context_size(self, att_context_size):
        if att_context_size not in self.att_context_size_all:
            logging.warning(
                f"att_context_size={att_context_size} is not among the list of the supported look-aheads: {self.att_context_size_all}"
            )
        if att_context_size is not None:
            self.att_context_size = att_context_size

        self.setup_streaming_params()

    def setup_streaming_params(
        self,
        chunk_size: int = None,
        shift_size: int = None,
        left_chunks: int = None,
        att_context_size: list = None,
        max_context: int = 10000,
    ):
        # This function is also used for FC models
        """
        This function sets the needed values and parameters to perform streaming. The configuration would be stored in self.streaming_cfg.
        The streaming configuration is needed to simulate streaming inference.

        Args:
            chunk_size (int): overrides the chunk size
            shift_size (int): overrides the shift size for chunks
            left_chunks (int): overrides the number of left chunks visible to each chunk
            max_context (int): the value used for the cache size of last_channel layers if left context is set to infinity (-1)
                Defaults to -1 (means feat_out is d_model)
        """
        streaming_cfg = CacheAwareStreamingConfig()
        # When att_context_size is not specified, it uses the default_att_context_size
        if att_context_size is None:
            att_context_size = self.att_context_size

        if chunk_size is not None:
            if chunk_size < 1:
                raise ValueError("chunk_size needs to be a number larger or equal to one.")
            lookahead_steps = chunk_size - 1
            streaming_cfg.cache_drop_size = chunk_size - shift_size
        elif self.att_context_style == "chunked_limited":
            lookahead_steps = att_context_size[1]
            streaming_cfg.cache_drop_size = 0
        elif self.att_context_style == "regular":
            lookahead_steps = att_context_size[1] * self.layers #+ self.conv_context_size[1] * self.n_layers
            streaming_cfg.cache_drop_size = lookahead_steps
        else:
            streaming_cfg.cache_drop_size = 0
            lookahead_steps = None
        if chunk_size is None:
            streaming_cfg.last_channel_cache_size = att_context_size[0] if att_context_size[0] >= 0 else max_context
        else:
            if left_chunks is None:
                raise ValueError("left_chunks can not be None when chunk_size is set.")
            streaming_cfg.last_channel_cache_size = left_chunks * chunk_size

        
        if hasattr(self.pre_encode, "get_sampling_frames"):
            sampling_frames = self.pre_encode.get_sampling_frames()
        else:
            sampling_frames = 0

        if isinstance(sampling_frames, list):
            streaming_cfg.chunk_size = [
                sampling_frames[0] + self.subsampling_factor * lookahead_steps,
                sampling_frames[1] + self.subsampling_factor * lookahead_steps,
            ]
        else:
            streaming_cfg.chunk_size = sampling_frames * (1 + lookahead_steps)
        if isinstance(sampling_frames, list):
            streaming_cfg.shift_size = [
                sampling_frames[0] + sampling_frames[1] * (lookahead_steps - streaming_cfg.cache_drop_size),
                sampling_frames[1] + sampling_frames[1] * (lookahead_steps - streaming_cfg.cache_drop_size),
            ]
        else:
            streaming_cfg.shift_size = sampling_frames * (1 + lookahead_steps - streaming_cfg.cache_drop_size)

        if isinstance(streaming_cfg.shift_size, list):
            streaming_cfg.valid_out_len = (
                streaming_cfg.shift_size[1] - sampling_frames[1]
            ) // self.subsampling_factor + 1
        else:
            streaming_cfg.valid_out_len = streaming_cfg.shift_size // self.subsampling_factor

        if hasattr(self.pre_encode, "get_streaming_cache_size"):
            streaming_cfg.pre_encode_cache_size = self.pre_encode.get_streaming_cache_size()
        else:
            streaming_cfg.pre_encode_cache_size = 0
        if isinstance(streaming_cfg.pre_encode_cache_size, list):
            if streaming_cfg.pre_encode_cache_size[1] >= 1:
                streaming_cfg.drop_extra_pre_encoded = (
                    1 + (streaming_cfg.pre_encode_cache_size[1] - 1) // self.subsampling_factor
                )
            else:
                streaming_cfg.drop_extra_pre_encoded = 0
        else:
            streaming_cfg.drop_extra_pre_encoded = streaming_cfg.pre_encode_cache_size // self.subsampling_factor

        for block in self.ngpt.transformer["h"]:
            block.cache_drop_size = streaming_cfg.cache_drop_size
        self.streaming_cfg = streaming_cfg   
    


    def _calc_context_sizes(
        self, att_context_size, att_context_probs, att_context_style,
    ):
        # convert att_context_size to a standard list of lists
        if att_context_size:
            att_context_size_all = list(att_context_size)
            if isinstance(att_context_size_all[0], int):
                att_context_size_all = [att_context_size_all]
            for i, att_cs in enumerate(att_context_size_all):
                if isinstance(att_cs, ListConfig):
                    att_context_size_all[i] = list(att_cs)
                if att_context_style == "chunked_limited":
                    if att_cs[0] > 0 and att_cs[0] % (att_cs[1] + 1) > 0:
                        raise ValueError(f"att_context_size[{i}][0] % (att_context_size[{i}][1] + 1) should be zero!")
                    if att_cs[1] < 0 and len(att_context_size_all) <= 1:
                        raise ValueError(
                            f"Right context (att_context_size[{i}][1]) can not be unlimited for chunked_limited style!"
                        )
        else:
            att_context_size_all = [[-1, -1]]

        if att_context_probs:
            if len(att_context_probs) != len(att_context_size_all):
                raise ValueError("The size of the att_context_probs should be the same as att_context_size.")
            att_context_probs = list(att_context_probs)
            if sum(att_context_probs) != 1:
                raise ValueError(
                    "The sum of numbers in att_context_probs should be equal to one to be a distribution."
                )
        else:
            att_context_probs = [1.0 / len(att_context_size_all)] * len(att_context_size_all)


        return att_context_size_all, att_context_size_all[0], att_context_probs, None
    
    def forward_for_export(
        self, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        raise NotImplementedError()

    def streaming_post_process(self, rets, keep_all_outputs=True):

        if len(rets) == 2:
            return rets[0], rets[1], None, None, None
        # keeping cache_last_time for support current cache-aware streaming
        (encoded, encoded_len, cache_last_channel_next, cache_last_time_next, cache_last_channel_next_len) = rets

        if cache_last_channel_next is not None and self.streaming_cfg.last_channel_cache_size >= 0:
            if self.streaming_cfg.last_channel_cache_size > 0:
                cache_last_channel_next = cache_last_channel_next[
                    :, :, -self.streaming_cfg.last_channel_cache_size :, :
                ]

        if self.streaming_cfg.valid_out_len > 0 and (not keep_all_outputs or self.att_context_style == "regular"):
            encoded = encoded[:, :, : self.streaming_cfg.valid_out_len]
            encoded_len = torch.clamp(encoded_len, max=self.streaming_cfg.valid_out_len)

        return (encoded, encoded_len, cache_last_channel_next, cache_last_time_next, cache_last_channel_next_len)

    #@typecheck()
    def forward(
        self, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        #cache_last_channel=None
        return self.forward_internal(
            audio_signal,
            length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )
    
    def forward_internal(
        self, audio_signal, length, cache_last_channel=None, cache_last_time=None, cache_last_channel_len=None
    ):
        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64, device=audio_signal.device
            )
        if self.training and len(self.att_context_size_all) > 1:
            cur_att_context_size = random.choices(self.att_context_size_all, weights=self.att_context_probs)[0]
        else:
            cur_att_context_size = self.att_context_size

        audio_signal = audio_signal.transpose(1, 2)
        x, length = self.pre_encode(x=audio_signal, lengths=length)
        
        length = length.to(torch.int64)
        if self.streaming_cfg.drop_extra_pre_encoded > 0 and cache_last_channel is not None:
            x = x[:, self.streaming_cfg.drop_extra_pre_encoded :, :]
            length = (length - self.streaming_cfg.drop_extra_pre_encoded).clamp(min=0)

        max_audio_length = x.size(1)

        if cache_last_channel is not None:
            cache_len = self.streaming_cfg.last_channel_cache_size
            cache_keep_size = max_audio_length - self.streaming_cfg.cache_drop_size
            max_audio_length = max_audio_length + cache_len
        
            padding_length = length + cache_len

            offset = torch.neg(cache_last_channel_len) + cache_len
            border = offset[0].item()
        else:
            padding_length = length
            cache_last_channel_next = None
            cache_len = 0
            offset = None 
            border = None
            cache_keep_size = None
        

        pad_mask = torch.arange(0, max_audio_length, device=x.device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)

        if cache_last_channel is not None:
            pad_mask = pad_mask[:, cache_len:]
                
        if self.att_context_style == "regular":
            block_mask = None
        
        elif self.att_context_style == "chunked_limited":
            # Currently implemented with left and righ
            # t context limited chunks

            # if cur_att_context_size[1] == -1:
            #     if cur_att_context_size[0] >= 0:
            #         att_mask = att_mask.triu(diagonal=-cur_att_context_size[0])
            # else:
            chunk_size = cur_att_context_size[1] + 1
            # left_chunks_num specifies the number of chunks to be visible by each chunk on the left side
            if cur_att_context_size[0] >= 0:
                left_chunks_num = cur_att_context_size[0] // chunk_size
            else:
                left_chunks_num = 10000
            
            # Concatenate the True values with the original tensor
            
        
            #Function that creates block_mask for flex attention, This function support out chunked logic
                
            def chunked_causal_mask(b, h, q_idx, kv_idx):
                """
                Implements a chunked causal attention mask at an index level.

                Args:
                    q_idx (int or Tensor): Query index.
                    kv_idx (int or Tensor): Key/Value index.

                Returns:
                    bool or Tensor: True if q_idx can attend to kv_idx, otherwise False.
                """
                pad_cur = pad_mask[b]
                q_chunk_idx = q_idx // chunk_size
                k_chunk_idx = kv_idx // chunk_size

                # Apply the chunked limited mask logic
                diff_chunks = q_chunk_idx - k_chunk_idx
                return (0 <= diff_chunks) & (diff_chunks <= left_chunks_num) & pad_cur[q_idx] & pad_cur[kv_idx]

            def chunked_causal_mask_inference(b, h, q_idx, kv_idx):
                """
                Implements a chunked causal attention mask at an index level.

                Args:
                    q_idx (int or Tensor): Query index.
                    kv_idx (int or Tensor): Key/Value index.

                Returns:
                    bool or Tensor: True if q_idx can attend to kv_idx, otherwise False.
                """
                num_true = max_audio_length - border - len(pad_mask[b])
                true_values = torch.ones(num_true, dtype=torch.bool,device=pad_mask.device)

                # Concatenate the True values with the original tensor
               
                pad_cur = pad_mask[b]
                new_tensor = torch.cat([true_values, pad_cur])
                
                return  pad_cur[q_idx] & new_tensor[kv_idx]
            
          
            if cache_last_channel is not None:
                block_mask =  create_block_mask(chunked_causal_mask_inference, B=x.shape[0], H=8, Q_LEN=max_audio_length-cache_len, KV_LEN=max_audio_length-border)
            else:
                block_mask =  create_block_mask(chunked_causal_mask, B=x.shape[0], H=8, Q_LEN=max_audio_length, KV_LEN=max_audio_length)
        
        if cache_last_channel is not None:
            cache_last_channel_next = []
        x = self.ngpt(x, block_mask, cache_last_channel=cache_last_channel, offset=border)
        

        if cache_last_channel is not None:
        
            audio_signal, cache_last_channel_next =  x
            return (
                audio_signal.transpose(1, 2),
                length,
                cache_last_channel_next,
                None, # for cache_time = convolution
                torch.clamp(cache_last_channel_len + cache_keep_size, max=cache_len),
            )
        else:
            x = x.transpose(1, 2)
            return x, length


    def normalize_matrices(self):
        if hasattr(self.pre_encode, "normalize_matrices"):
            self.pre_encode.normalize_matrices()
        self.ngpt.normalize_matrices()


# def apply_rotary_position_embeddings(sinusoidal_pos, q, k):
#     # Ensure q and k have correct positional embeddings
#     sin_q, cos_q = sinusoidal_pos[-q.shape[2]:].chunk(2, dim=-1)  # Select for q length
#     sin_k, cos_k = sinusoidal_pos[-k.shape[2]:].chunk(2, dim=-1)  # Select for k length

#     # Apply rotary embeddings
#     def apply_rotary(q, sin, cos):
#         q_rot = torch.stack((-q[..., 1::2], q[..., ::2]), dim=-1)
#         q_rot = torch.reshape(q_rot, q.shape[:-1] + (q.shape[-1] // 2, 2)) * torch.stack((cos, sin), dim=-1)
#         return torch.reshape(q_rot, q.shape)

#     q_rot = apply_rotary(q, sin_q, cos_q)
#     k_rot = apply_rotary(k, sin_k, cos_k)

#     return q_rot.to(q.dtype), k_rot.to(k.dtype)


# def get_sinusoidal_embeddings(n_positions, dim, device):
#     """Generate sinusoidal positional embeddings."""
#     position = torch.arange(n_positions, dtype=torch.float, device=device).unsqueeze(1)
#     div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / dim))
#     sinusoidal_emb = torch.empty((n_positions, dim), device=device)
#     sinusoidal_emb[:, 0::2] = torch.sin(position * div_term)
#     sinusoidal_emb[:, 1::2] = torch.cos(position * div_term)
#     return sinusoidal_emb


def justnorm(x, fp32: bool = False, idim: int = -1,eps: float = 1e-10):
    
    if fp32:
        dtype = x.dtype
        x = x.float()
        res = (x / x.norm(p=2, dim=idim, keepdim=True)).to(dtype=dtype)
    else:
        norm = x.norm(p=2, dim=idim, keepdim=True)
        res = x / (norm + eps)
    return res


def justnorm_fp32(x, idim: int = -1):
    return justnorm(x, idim=idim, fp32=True)


class Block(nn.Module):

    def __init__(self, config, iblock):
        super().__init__()
        self.config = config

        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.att_c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.rotary_emb = RotaryEmbedding(dim =config.n_embd // config.n_head)
        self.c_fc = nn.Linear(config.n_embd, 2 * 4 * config.n_embd, bias=config.bias)
        self.silu = nn.SiLU()
        self.mlp_c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.offset = 0
        if config.use_nGPT == 0:
            self.rmsnorm_att = RMSNorm(config.n_embd)
            self.rmsnorm_mlp = RMSNorm(config.n_embd)

        if config.use_nGPT == 1:
            self.attn_alpha_init_value = 0.05
            self.attn_alpha_init_scaling = config.base_scale
            self.attn_alpha = torch.nn.Parameter(
                self.attn_alpha_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32)
            )

            self.mlp_alpha_init_value = 0.05
            self.mlp_alpha_init_scaling = config.base_scale
            self.mlp_alpha = torch.nn.Parameter(
                self.mlp_alpha_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32)
            )

            self.sqk_init_value = 1.0
            self.sqk_init_scaling = config.base_scale
            self.sqk = torch.nn.Parameter(self.sqk_init_scaling * torch.ones(self.config.n_embd, dtype=torch.float32))

            self.suv_init_value = 1.0
            self.suv_init_scaling = 1.0
            self.suv = torch.nn.Parameter(
                self.suv_init_scaling * torch.ones(2 * 4 * config.n_embd, dtype=torch.float32)
            )


    def forward(self, h, block_mask=None,cache_last_channel=None, offset=None):
        B, T, C = h.size()
        key, value, query, cache = self.update_cache(key=h,value=h,query=h,cache=cache_last_channel)

       # hin = h
       # if self.config.use_nGPT == 0:
       #     q = k = v = self.rmsnorm_att(h)
        
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        #using to ignore part of the cache that is all zeros 
        if offset:
            k  = k[:, offset:, :]
            v = v[:, offset:, :]
        if offset == 0:
            self.offset += 14

        q = q.view(B, T, self.config.n_head, self.config.n_embd // self.config.n_head)
        k = k.view(B, k.shape[1], self.config.n_head, self.config.n_embd // self.config.n_head)   
        v = v.view(B, k.shape[1], self.config.n_head, self.config.n_embd // self.config.n_head)

        
        q = q.transpose(2, 1)
        k = k.transpose(2, 1)


        if offset is None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)
        else:
            q = self.rotary_emb.rotate_queries_or_keys(q, offset=k.shape[2] - q.shape[2] + self.offset)
            k = self.rotary_emb.rotate_queries_or_keys(k, offset=self.offset)
        q = q.transpose(2, 1)
        k = k.transpose(2, 1)

        if self.config.use_nGPT == 1:
            sqk = (self.sqk * (self.sqk_init_value / self.sqk_init_scaling)).view(
                1, 1, self.config.n_head, self.config.n_embd // self.config.n_head
            )
            q = sqk * justnorm(q)
            k = sqk * justnorm(k)
        sqrt_head_dim = (self.config.n_embd / self.config.n_head) ** 0.5
        if self.config.use_nGPT == 0:
            softmax_scale = 1.0 / sqrt_head_dim
        if self.config.use_nGPT == 1:
            softmax_scale = sqrt_head_dim
        if block_mask is None:
            y = flash_attn_func(
                q.to(torch.bfloat16),
                k.to(torch.bfloat16),
                v.to(torch.bfloat16),
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=False,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=False,
            )
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            y = flex_attention(
                q.to(torch.bfloat16),
                k.to(torch.bfloat16),
                v.to(torch.bfloat16),
                block_mask=block_mask,
                scale=softmax_scale,
            )
        y = y.to(dtype=q.dtype)
        
        y = y.transpose(1, 2).contiguous().view(B, T, self.config.n_embd)

        h_att = self.att_c_proj(y)

        if self.config.use_nGPT == 0:
            h = h + h_att
        if self.config.use_nGPT == 1:
            lr = self.attn_alpha * (self.attn_alpha_init_value / self.attn_alpha_init_scaling)
            lr = torch.abs(lr)

            A_norm = justnorm(h)  # normally, normalization is not needed
            B_norm = justnorm(h_att)

            # res = (1.0 - lr) * A_norm + lr * B_norm
            res = A_norm + lr * (B_norm - A_norm)
            h = justnorm(res)

        hin = h
        if self.config.use_nGPT == 0:
            hin = self.rmsnorm_mlp(h)
        uv = self.c_fc(hin)
        if self.config.use_nGPT == 1:
            suv = self.suv * ((self.suv_init_value / self.suv_init_scaling) * (self.config.n_embd**0.5))
            uv = suv * uv
        u, v = torch.chunk(uv, 2, dim=-1)
        x_mlp = u * self.silu(v)
        h_mlp = self.mlp_c_proj(x_mlp)

        if self.config.use_nGPT == 0:
            h = h + h_mlp
        if self.config.use_nGPT == 1:
            lr = self.mlp_alpha * (self.mlp_alpha_init_value / self.mlp_alpha_init_scaling)
            lr = torch.abs(lr)

            A_norm = justnorm(h)  # normally, normalization is not needed
            B_norm = justnorm(h_mlp)

            # res = (1.0 - lr) * A_norm + lr * B_norm
            res = A_norm + lr * (B_norm - A_norm)
            h = justnorm(res)

        if cache is None:
            return h
        else:
            return h, cache, None

    def update_cache(self, key, value, query, cache):
        if cache is not None:
            key = value = torch.cat([cache, key], dim=1)
            q_keep_size = query.shape[1] -  self.cache_drop_size
            cache = torch.cat([cache[:, q_keep_size:, :], query[:, :q_keep_size, :]], dim=1)
        return key, value, query, cache




@dataclass
class GPTConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 1024
    base_scale: float = 1.0 / (1024.0**0.5)  # 1 / sqrt(n_embd)
    use_nGPT: int = 0
    dropout: float = 0.0
    bias: bool = False
    att_context_size: int = 0


class RMSNorm(torch.nn.Module):
    def __init__(self, embdim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(embdim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = torch.mean(x * x, dim=-1, keepdim=True)
        xnorm = x * torch.rsqrt(norm + self.eps)
        xnorm = xnorm.to(dtype=dtype)
        return xnorm * self.weight


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                # wte=nn.Embedding(config.vocab_size, config.n_embd),
                # drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config, il) for il in range(config.n_layer)])
            )
        )
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # *we don't use it becuase in the nGPT paper there was no weight tying of weights*
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=config.base_scale / math.sqrt(2 * config.n_layer))
        # report number of parameters
        logging.info("[nGPT] number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

        if config.use_nGPT == 0:
            self.rmsnorm_f = RMSNorm(config.n_embd)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #    n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_scale)

    def forward(self, x, block_mask=None, cache_last_channel=None, offset=None):
            
        cache_last_channel_next = []
        audio_signal = x
        for idx, block in enumerate(self.transformer.h):

            if cache_last_channel is not None:            
                cache_last_channel_cur = cache_last_channel[idx]
            else:
                cache_last_channel_cur = None
            x = block(
                audio_signal, 
                block_mask=block_mask,
                cache_last_channel=cache_last_channel_cur,
                offset=offset
                )
            if cache_last_channel_cur is not None:
                (audio_signal, cache_last_channel_cur, _) = x
                cache_last_channel_next.append(cache_last_channel_cur)
            else:
                audio_signal = x
        # if self.config.use_nGPT == 0:
        #     x = self.rmsnorm_f(x)
        if cache_last_channel is not None:
            cache_last_channel_next = torch.stack(cache_last_channel_next, dim=0)

            return (
                audio_signal,
                cache_last_channel_next,
            )
        else:
            return x

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logging.info(
            f"[nGPT] num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        logging.info(
            f"[nGPT] num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        use_fused = False  # fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        logging.info(f"[nGPT] using fused AdamW: {use_fused}")
        return optimizer

    def normalize_matrices(self):
        if not self.config.use_nGPT:
            return

        transformer = self.transformer
        module = self

        for layer_idx in range(0, module.config.n_layer):
            block = transformer["h"][layer_idx]

            block.query.weight.data.copy_(justnorm_fp32(block.query.weight.data, 1))  # n_proj, n_embd
            block.key.weight.data.copy_(justnorm_fp32(block.key.weight.data, 1))  # n_proj, n_embd
            block.value.weight.data.copy_(justnorm_fp32(block.value.weight.data, 1))  # n_proj, n_embd
            block.att_c_proj.weight.data.copy_(justnorm_fp32(block.att_c_proj.weight.data, 0))  # n_embd, n_proj

            block.c_fc.weight.data.copy_(justnorm_fp32(block.c_fc.weight.data, 1))  # n_proj, n_embd
            block.mlp_c_proj.weight.data.copy_(justnorm_fp32(block.mlp_c_proj.weight.data, 0))  # n_embd, n_proj


class NGPTHead(NeuralModule):
    def __init__(
        self,
        feat_in: int,
        num_classes: int,
        base_scale: float = 1 / (1024**0.5),
        use_log_softmax: bool = True,
        include_blank: bool = True,
        vocabulary=None,  # ignored, included for compatibility
    ) -> None:
        super().__init__()
        self._num_classes = num_classes
        if include_blank:
            self._num_classes += 1
        self.lm_head = nn.Linear(feat_in, self._num_classes, bias=False)
        self.sz_init_scaling = base_scale
        self.sz = torch.nn.Parameter(self.sz_init_scaling * torch.ones(self._num_classes, dtype=torch.float32))
        self.use_log_softmax = use_log_softmax
        self.vocabulary = vocabulary

    def _init_weights(self):
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.base_scale)

    def normalize_matrices(self):
        self.lm_head.weight.data.copy_(justnorm_fp32(self.lm_head.weight.data, 1))

    def forward(self, encoder_output):
        x = encoder_output.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        logits = self.lm_head(x)
        sz = self.sz * (1.0 / self.sz_init_scaling)
        logits = sz * logits
        if self.use_log_softmax:
            logits = nn.functional.log_softmax(logits, dim=-1)
        return logits

    @property
    def num_classes_with_blank(self):
        return self._num_classes


class NGPTStackingSubsampling(torch.nn.Module):
    """Stacking subsampling which simply stacks consecutive frames to reduce the sampling rate
    Args:
        subsampling_factor (int): The subsampling factor
        feat_in (int): size of the input features
        feat_out (int): size of the output features
    """

    def __init__(
        self,
        subsampling_factor: int,
        feat_in: int,
        feat_out: int,
        use_bias: bool = False,
        base_scale: float = 1 / (1024**0.5),
    ):
        super().__init__()
        self.subsampling_factor = subsampling_factor
        self.proj_out = torch.nn.Linear(subsampling_factor * feat_in, feat_out, bias=use_bias)
        self.pad_frame = nn.Parameter(torch.ones(feat_in, dtype=torch.float32))

    def _init_weights(self):
        torch.nn.init.normal_(self.proj_out.weight, mean=0.0, std=self.config.base_scale)
        if self.proj_out.bias is not None:
            torch.nn.init.zeros_(self.proj_out.bias)

    def normalize_matrices(self):
        self.proj_out.weight.data.copy_(justnorm_fp32(self.proj_out.weight.data, 0))
    def get_sampling_frames(self):
        return self.subsampling_factor

    def get_streaming_cache_size(self):
        return 0

    def forward(self, x, lengths):
        b, t, h = x.size()
        # To consider correct pad_size in a batch for lengths
        sample_pad_sizes = []
        remainder = lengths % self.subsampling_factor
        sample_pad_sizes = (self.subsampling_factor - remainder) % self.subsampling_factor

        pad_size = (self.subsampling_factor - (t % self.subsampling_factor)) % self.subsampling_factor
        lengths = torch.div(lengths + sample_pad_sizes, self.subsampling_factor, rounding_mode='floor')
        # Pad and fill padding frames (all-zero) with a learnable padding 'embedding'
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_size))
        x[(x == 0).all(dim=-1)] = self.pad_frame

        _, t, _ = x.size()
        x = torch.reshape(x, (b, t // self.subsampling_factor, h * self.subsampling_factor))
        x = justnorm(x)
        x = self.proj_out(x)
        x = justnorm(x)
        return x, lengths